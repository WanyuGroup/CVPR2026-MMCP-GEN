This repository provides a multimodal conditional training framework for protein sequence generation.  
It focuses on reusable **MMCP (Multi-Modal Conditional Prompting)** modules and an end-to-end training template.

---

## Overview

This project extends DPLM with multimodal conditions (e.g., structure, ligand, function annotations, text) and injects them into the language model through lightweight plug-in components.

The two key files are:

- `mmcp_modules.py`: reusable MMCP core modules
- `Train_mmcp_gen.py`: end-to-end MMCP-GEN training template

---

## Repository Scope


- `src/`, `scripts/`, `analysis/`, `assets/`, `vendor/`
- `mmcp_modules.py`
- `Train_mmcp_gen.py` (or your actual training entry)
- `multimodal_encoders.py`
- training/inference scripts such as `train_mmcp.py`, `train_multimodal_dplm.py`, `generate_dplm.py`
- `requirements.txt`, `env.yml`, `setup.py`, `setup.cfg`


---

## Core Modules: `mmcp_modules.py`

`mmcp_modules.py` includes four key building blocks:

1. **ModalityIndicatorHead**
   - Prepends one learnable indicator token to each modality token sequence.
   - Purpose: explicitly mark modality identity and reduce cross-modality confusion.

2. **MMCP_LQ (Learnable Query Fusion)**
   - Concatenates multimodal tokens with shared/modality-specific queries.
   - Uses a Transformer Encoder to fuse them and outputs fused query tokens `Q_out`.

3. **CrossAttentionAdapter**
   - Performs lightweight cross-attention between LM hidden states `H` and condition summary `S`.
   - Produces residual delta `delta` (with learnable scaling) to inject into backbone layers.

4. **Adapter Attachment Utilities**
   - `attach_adapters_to_modules(...)`: attaches adapters to target layers.
   - `register_adapter_hooks(...)`: registers forward hooks for non-invasive condition injection.

Design goal: **add multimodal controllability to the original DPLM backbone**.

---

## Training Entry: `Train_mmcp_gen.py`

`Train_mmcp_gen.py` is an MMCP-GEN training template with a full pipeline:

1. Load DPLM (prefer local `DPLM_650M`, fallback to Hugging Face, then fallback dummy model)
2. Initialize multimodal encoders (GVP-Transformer / SchNet / ProtBERT, configurable)
3. Build MMCP components (indicator, queries, fusion, adapters, projection heads)
4. Register adapter hooks on selected Transformer layers
5. Jointly optimize:
   - sequence cross-entropy loss `L_CE`
   - structural alignment loss `L_align` (InfoNCE)
6. Save MMCP parameters and encoder states

---

## Quick Start

### 1) Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or with Conda:

```bash
conda env create -f env.yml
conda activate dplm
```

### 2) Run MMCP-GEN Training

```bash
python Train_mmcp_gen.py \
  --epochs 1 \
  --batch_size 8 \
  --lr 1e-5 \
  --K_shared 16 \
  --K_per_modality 2 \
  --gamma 1.0 \
  --save_path mmcp_gen_checkpoint.pt
```

If using Inhouse data:

```bash
python Train_mmcp_gen.py \
  --use_inhouse \
  --inhouse_data_path /path/to/your/inhouse.jsonl \
  --max_seq_len 256 \
  --structure_dim 128 \
  --batch_size 8 \
  --epochs 5
```

---

## Important Arguments (`Train_mmcp_gen.py`)

- `--K_shared`: number of shared queries
- `--K_per_modality`: number of modality-specific queries per modality
- `--adapter_n_heads`: attention heads used in adapters
- `--gamma`: weight for alignment loss
- `--modality_dropout_prob`: modality dropout probability
- `--disable_*_encoder` / `--freeze_*`: enable/freeze each modality encoder

---

## Training Outputs

Default checkpoint output:

- `mmcp_gen_checkpoint.pt`

Typically contains:

- MMCP fusion module weights
- modality indicator weights
- adapter weights
- condition projection weights
- optional multimodal encoder weights

---


## Acknowledgements

- DPLM-related open implementations and community work
- PyTorch ecosystem for multimodal modeling components

If you build on this repository, citations and references are welcome.

# MMCP-GEN

一个面向蛋白序列生成的多模态条件训练仓库。  
本仓库重点提供了 **MMCP（Multi-Modal Conditional Prompting）** 的可复用模块，以及一个可直接运行/改造的训练脚本模板。

---

## 项目定位

该仓库用于在 DPLM 基础上引入多模态条件（如结构、配体、功能注释、文本），通过轻量模块把条件信息注入到语言模型中，从而提升条件可控生成能力。

核心代码主要在两个文件：

- `mmcp_modules.py`：MMCP 核心模块定义（可独立复用）
- `Train_mmcp_gen.py`：MMCP-GEN 训练脚本模板（端到端训练流程）

---

## 代码结构

- `src/`, `scripts/`, `analysis/`, `assets/`, `vendor/`
- `mmcp_modules.py`
- `Train_mmcp_gen.py`（或你实际训练入口脚本）
- `multimodal_encoders.py`
- `train_mmcp.py`, `train_multimodal_dplm.py`, `generate_dplm.py` 等训练/推理脚本
- `requirements.txt`, `env.yml`, `setup.py`, `setup.cfg`

---

## 核心模块说明：`mmcp_modules.py`

`mmcp_modules.py` 提供了 MMCP 方案的 4 类关键能力：

1. **ModalityIndicatorHead**
   - 为每个模态 token 序列前拼接一个可学习 indicator token。
   - 作用：显式标识 token 来源模态，减少模态混淆。

2. **MMCP_LQ（Learnable Query Fusion）**
   - 将拼接后的多模态 token 与共享/模态专属查询一起输入 Transformer Encoder 融合。
   - 输出融合后的查询向量 `Q_out`，作为条件摘要。

3. **CrossAttentionAdapter**
   - 对语言模型隐藏状态 `H` 与条件摘要 `S` 做轻量 cross-attention。
   - 输出残差增量 `delta`，通过可学习缩放参数注入主干模型层。

4. **attach/register 工具函数**
   - `attach_adapters_to_modules(...)`：给目标层挂载 adapter。
   - `register_adapter_hooks(...)`：通过 forward hook 非侵入注入条件信息。

这套设计的目标是：**改动原始 DPLM 主干，实现可插拔式多模态条件训练**。

---

## 训练入口说明：`Train_mmcp_gen.py`

`Train_mmcp_gen.py` 是一个 MMCP-GEN 训练模板，包含完整流程：

1. 加载 DPLM（优先本地 `DPLM_650M`，失败时尝试 Hugging Face，再失败则用占位模型）
2. 初始化多模态编码器（GVP-Transformer / SchNet / ProtBERT，可开关）
3. 构建 MMCP 组件（indicator、query、fusion、adapter、投影头）
4. 在选定 Transformer 层注册 adapter hook
5. 训练时联合优化：
   - 序列交叉熵损失 `L_CE`
   - 结构对齐损失 `L_align`（InfoNCE）
6. 保存 MMCP 相关参数与编码器状态

---

## 快速开始

### 1) 环境安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

或使用 Conda：

```bash
conda env create -f env.yml
conda activate dplm
```

### 2) 运行 MMCP-GEN 训练模板

```bash
python Train_mmcp_gen.py \
  --epochs 1 \
  --batch_size 8 \
  --lr 1e-5 \
  --K_shared 16 \
  --K_per_modality 2 \
  --gamma 1.0 \
  --save_path mmcp_gen_checkpoint.pt
```

如果使用 Inhouse 数据集：

```bash
python Train_mmcp_gen.py \
  --use_inhouse \
  --inhouse_data_path /path/to/your/inhouse.jsonl \
  --max_seq_len 256 \
  --structure_dim 128 \
  --batch_size 8 \
  --epochs 5
```

---

## 关键参数（Train_mmcp_gen.py）

- `--K_shared`：共享 query 数量
- `--K_per_modality`：每个模态专属 query 数量
- `--adapter_n_heads`：adapter 的注意力头数
- `--gamma`：对齐损失权重
- `--modality_dropout_prob`：模态 dropout 概率
- `--disable_*_encoder` / `--freeze_*`：各模态编码器的启用与冻结控制

---

## 训练输出

默认保存文件为：

- `mmcp_gen_checkpoint.pt`

其中包含：

- MMCP fusion 模块参数
- 模态 indicator 参数
- adapter 参数
- 条件投影参数
- （可选）多模态编码器参数

---


## 致谢

- DPLM 相关工作与社区实现
- PyTorch 生态中的多模态建模组件

如你基于本仓库做二次开发，欢迎在论文或项目说明中引用本仓库思路与实现。
