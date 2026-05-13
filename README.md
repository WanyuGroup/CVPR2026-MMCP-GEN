<img width="5000" height="2044" alt="all_1_new" src="https://github.com/user-attachments/assets/5ac7efb8-f30a-4a66-aeaf-9f65577217ef" />
<img width="5238" height="2117" alt="all_2" src="https://github.com/user-attachments/assets/fbb59775-8aeb-4f79-9530-65a7e50ee86d" />

This repository provides a multimodal conditional training framework for protein sequence generation.  
It focuses on reusable **MMCP-GEN** modules and an end-to-end training template.

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
- `Train_mmcp_gen.py` 
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
5. optimize:
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
