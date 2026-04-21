# 多模态DPLM系统

本系统实现了支持多模态条件输入的蛋白质语言模型，能够同时利用3D结构、小分子配体和功能注释信息进行条件生成。

## 系统架构

### 多模态编码器
- **GVP-Transformer**: 编码蛋白质3D骨架信息
- **SchNet**: 编码小分子配体信息  
- **ProtBERT**: 编码功能注释信息（GO terms、IPR、EC等）

### 融合机制
- **MMCP-LQ**: 多模态条件投影学习查询融合
- **交叉注意力适配器**: 将条件信息注入到DPLM的transformer层
- **InfoNCE对齐损失**: 确保序列和结构表示的对齐

## 文件结构

```
├── multimodal_encoders.py      # 多模态编码器实现
├── multimodal_dataset.py       # 多模态数据加载器
├── multimodal_dplm.py          # 多模态DPLM模型
├── train_multimodal_dplm.py    # 训练脚本
├── example_usage.py            # 使用示例
└── MULTIMODAL_README.md        # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install torch torchvision torchaudio
pip install transformers
pip install esm
pip install numpy pandas
```

### 2. 运行示例

```bash
python example_usage.py
```

这将创建示例数据并测试所有组件。

### 3. 训练模型

```bash
python train_multimodal_dplm.py \
    --train_data_path data/train_multimodal.jsonl \
    --val_data_path data/val_multimodal.jsonl \
    --epochs 10 \
    --batch_size 8 \
    --learning_rate 1e-4
```

## 数据格式

### 输入数据格式（JSONL）

每行应包含以下字段：

```json
{
    "name": "sample_001",
    "sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    "structure": {
        "ca_coords": [[x1, y1, z1], [x2, y2, z2], ...]
    },
    "ligand": {
        "atoms": ["C", "N", "O", "H", ...],
        "coords": [[x1, y1, z1], [x2, y2, z2], ...]
    },
    "annotations": {
        "go_terms": ["GO:0008150", "GO:0003674"],
        "ipr": ["IPR000001", "IPR000002"],
        "ec": ["EC:1.1.1.1", "EC:2.7.1.1"],
        "sequence_annotation": "Protein function description"
    },
    "metadata": {
        "source": "PDB",
        "length": 67,
        "ligand_atoms": 25
    }
}
```

### 字段说明

- **sequence**: 蛋白质氨基酸序列
- **structure.ca_coords**: CA原子坐标列表
- **ligand.atoms**: 配体原子类型列表
- **ligand.coords**: 配体原子坐标列表
- **annotations**: 功能注释信息
  - **go_terms**: GO功能注释
  - **ipr**: InterPro注释
  - **ec**: EC编号
  - **sequence_annotation**: 序列功能描述

## 模型配置

### 多模态编码器配置

```python
multimodal_config = {
    'backbone_dim': 256,      # 骨架编码维度
    'ligand_dim': 256,        # 配体编码维度
    'function_dim': 256,      # 功能注释编码维度
    'output_dim': 256,        # 输出维度
    'fusion_method': 'concat' # 融合方法：'concat' 或 'attention'
}
```

### 训练参数

```python
training_args = {
    'epochs': 10,             # 训练轮数
    'batch_size': 8,          # 批次大小
    'learning_rate': 1e-4,    # 学习率
    'weight_decay': 1e-5,     # 权重衰减
    'max_seq_len': 512,       # 最大序列长度
    'max_ligand_atoms': 200,  # 最大配体原子数
}
```

## 使用方法

### 1. 创建多模态编码器

```python
from multimodal_encoders import MultimodalConditionEncoder

encoder = MultimodalConditionEncoder(
    backbone_dim=256,
    ligand_dim=256,
    function_dim=256,
    output_dim=256,
    fusion_method="concat"
)
```

### 2. 创建多模态DPLM

```python
from multimodal_dplm import create_multimodal_dplm

model = create_multimodal_dplm(
    base_model=base_dplm_model,
    multimodal_config=multimodal_config
)
```

### 3. 前向传播

```python
logits = model(
    input_ids=protein_sequences,
    backbone_coords=structure_coords,
    ligand_atoms=ligand_atom_types,
    ligand_coords=ligand_coordinates,
    function_annotations=function_annotations
)
```

### 4. 计算损失

```python
losses = model.compute_loss(
    input_ids=input_ids,
    labels=labels,
    backbone_coords=backbone_coords,
    ligand_atoms=ligand_atoms,
    ligand_coords=ligand_coords,
    function_annotations=function_annotations
)
```

## 训练流程

### 1. 数据准备

```python
# 创建示例数据
from multimodal_dataset import create_sample_data

create_sample_data("data/train.jsonl", num_samples=1000)
create_sample_data("data/val.jsonl", num_samples=200)
```

### 2. 创建数据加载器

```python
from multimodal_dataset import create_multimodal_dataloader

train_loader = create_multimodal_dataloader(
    data_path="data/train.jsonl",
    batch_size=8,
    max_seq_len=512,
    max_ligand_atoms=200
)
```

### 3. 训练模型

```python
from train_multimodal_dplm import MultimodalDPLMTrainer

trainer = MultimodalDPLMTrainer(args)
trainer.train()
```

## 模型特点

### 1. 多模态融合
- 支持3D结构、小分子配体和功能注释的同时输入
- 使用MMCP-LQ机制进行多模态特征融合
- 通过交叉注意力将条件信息注入到语言模型

### 2. 条件生成
- 基于多模态条件进行蛋白质序列生成
- 支持结构约束、配体结合和功能导向的生成
- 保持与原始DPLM的兼容性

### 3. 对齐学习
- 使用InfoNCE损失确保序列和结构表示的对齐
- 支持结构-序列一致性约束
- 提高生成质量

## 性能优化

### 1. 内存优化
- 支持梯度检查点
- 使用混合精度训练
- 动态批次大小调整

### 2. 训练加速
- 支持多GPU训练
- 使用数据并行
- 优化数据加载

### 3. 模型压缩
- 支持模型量化
- 知识蒸馏
- 剪枝优化

## 故障排除

### 1. 常见问题

**Q: 内存不足怎么办？**
A: 减少批次大小、序列长度或配体原子数，使用梯度检查点。

**Q: 训练速度慢怎么办？**
A: 使用GPU训练，增加数据加载器工作进程数，使用混合精度。

**Q: 模型不收敛怎么办？**
A: 调整学习率，检查数据质量，增加训练轮数。

### 2. 调试技巧

```python
# 检查数据格式
for batch in dataloader:
    print(f"输入形状: {batch['input_ids'].shape}")
    print(f"结构形状: {batch['structure_coords'].shape}")
    break

# 检查模型输出
with torch.no_grad():
    output = model(input_ids, **conditions)
    print(f"输出形状: {output.shape}")
```

## 扩展功能

### 1. 自定义编码器
```python
class CustomEncoder(nn.Module):
    def forward(self, x):
        # 自定义编码逻辑
        return encoded_features

# 替换默认编码器
model.multimodal_encoder.custom_encoder = CustomEncoder()
```

### 2. 添加新模态
```python
# 添加新的模态编码器
model.add_modality_encoder('new_modality', CustomEncoder())
```

### 3. 自定义损失函数
```python
def custom_loss(logits, targets, conditions):
    # 自定义损失计算
    return loss

model.custom_loss_fn = custom_loss
```

## 引用

如果您使用了本系统，请引用相关论文：

```bibtex
@article{multimodal_dplm_2024,
  title={Multimodal Conditional Protein Language Model},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## 许可证

本项目采用Apache 2.0许可证。



---

**注意**: 本系统基于DPLM架构开发，需要确保基础DPLM模型正确加载。如果遇到问题，请检查DPLM模型路径和依赖项安装。





