#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py -- MMCP-GEN integrated training script (template)

功能概览：
- 集成 MMCP-IH（指示头）、MMCP-LQ（learnable query fusion）、CrossAttentionAdapter（注入到 DPLM）
- 在训练循环内对模态条件编码、拼接、融合、注入 DPLM；计算 CE 与结构对比损失（InfoNCE）
- 只更新指定的可训练参数（projection heads、indicator、MMCP-LQ、adapter、g_proj）
注意：请根据你工程中实际的变量名替换部分占位符（我在代码中用大写注释标注）
"""

import os
import math
import argparse
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# === 导入你项目中的 DPLM / encoders ===
# 下面是占位导入：请替换为你仓库中真实的 model/encoder 导入路径
# from your_project.model import build_model, load_model_weights
# from your_project.data import build_train_dataset, collate_fn
# from your_project.encoders import build_modality_encoders  # dict of modality->encoder

# 我们新建或使用之前添加的 mmcp_modules（确保 mmcp_modules.py 在 import 路径中）
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

try:
    from mmcp_modules import (
        ModalityIndicatorHead,
        MMCP_LQ,
        CrossAttentionAdapter,
        attach_adapters_to_modules,
        register_adapter_hooks,
    )
    from multimodal_encoders import (
        GVPTransformerEncoder,
        SchNetEncoder,
        ProtBERTEncoder,
    )
except Exception as e:
    raise ImportError("无法导入 mmcp_modules。请确保 mmcp_modules.py 在 PYTHONPATH 中。详细错误：" + str(e))

# 导入Inhouse数据集
try:
    from inhouse_dataset import InhouseDataset, create_inhouse_dataloader
except Exception as e:
    print(f"⚠️ 无法导入Inhouse数据集: {e}")
    InhouseDataset = None
    create_inhouse_dataloader = None


# ---------------------------
# Utilities
# ---------------------------
def info_nce_loss(p: torch.Tensor, z: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    p: (B, D) projected sequence embeddings
    z: (B, D) structural embeddings (from frozen GVP)
    Returns InfoNCE loss using in-batch negatives (assumes positive pairs align on batch index)
    """
    # normalize
    p_norm = p / p.norm(dim=1, keepdim=True).clamp(min=1e-8)
    z_norm = z / z.norm(dim=1, keepdim=True).clamp(min=1e-8)
    logits = (p_norm @ z_norm.t()) / temperature  # (B, B)
    labels = torch.arange(p.size(0), device=p.device)
    loss = F.cross_entropy(logits, labels)
    return loss


def find_transformer_layer_modules(model) -> List[nn.Module]:
    """
    Heuristic: try common attribute paths that hold transformer blocks.
    If fails, fallback to collecting modules whose class name contains 'Block'/'Layer'/'Transformer'.
    """
    candidate_paths = [
        'net.transformer.blocks',
        'transformer.blocks',
        'net.encoder.layers',
        'encoder.layers',
        'net.layers',
        'layers',
        'blocks',
    ]
    for p in candidate_paths:
        obj = model
        ok = True
        for part in p.split('.'):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok:
            if isinstance(obj, (list, tuple)):
                return list(obj)
            if hasattr(obj, '__iter__'):
                try:
                    return list(obj)
                except Exception:
                    pass
    # fallback: search by name
    candidates = []
    for name, mod in model.named_modules():
        cn = mod.__class__.__name__.lower()
        if 'block' in cn or 'layer' in cn or 'transformer' in cn:
            candidates.append(mod)
    # dedupe & return
    uniq = []
    for m in candidates:
        if m not in uniq:
            uniq.append(m)
    return uniq


# ---------------------------
# Main training function
# ---------------------------
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # --------
    # 1. load / build model and frozen encoders
    # --------
    # 尝试加载真实的 DPLM 模型
    print("正在尝试加载 DPLM 模型...")
    
    # 定义本地 checkpoint 路径
    import os
    local_checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DPLM_650M')
    
    try:
        # 方法1：尝试加载本地 DPLM 650M checkpoint
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
        
        from byprot.models.dplm.dplm import DiffusionProteinLanguageModel as DPLM
        
        # 检查本地 checkpoint 是否存在
        if os.path.exists(local_checkpoint_path):
            print(f"找到本地 DPLM 650M checkpoint: {local_checkpoint_path}")
            print("正在加载本地 DPLM 650M 模型...")
            model = DPLM.from_pretrained(local_checkpoint_path, from_huggingface=False)
            model = model.eval()
            model = model.to(device)
            print(f"✅ 本地 DPLM 650M 模型加载成功！模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        else:
            raise FileNotFoundError(f"本地 checkpoint 不存在: {local_checkpoint_path}")
        
    except Exception as e:
        print(f"❌ 加载本地 checkpoint 失败: {e}")
        print("尝试从 Hugging Face 加载...")
        
        try:
            # 方法2：尝试从 Hugging Face 加载
            from byprot.models.dplm.dplm import DiffusionProteinLanguageModel as DPLM
            
            print("正在从 Hugging Face 加载 DPLM 650M 模型...")
            model = DPLM.from_pretrained("airkingbd/dplm_650m")
            model = model.eval()
            model = model.to(device)
            print(f"✅ DPLM 650M 模型加载成功！模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
            
        except Exception as e2:
            print(f"❌ 从 Hugging Face 加载也失败: {e2}")
            print("尝试使用 transformers 直接加载...")
            
            try:
                # 方法3：使用 transformers 直接加载
                from transformers import AutoModel, AutoTokenizer
                
                print("正在使用 transformers 加载本地 DPLM 650M 模型...")
                tokenizer = AutoTokenizer.from_pretrained(local_checkpoint_path)
                model = AutoModel.from_pretrained(local_checkpoint_path)
                model = model.eval()
                model = model.to(device)
                
                # 为兼容性添加必要的属性
                # 保持原始配置，只添加必要的属性
                if not hasattr(model.config, '_attn_implementation'):
                    model.config._attn_implementation = "eager"
                model.tokenizer = tokenizer
                
                print(f"✅ 使用 transformers 加载成功！模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
                
            except Exception as e3:
                print(f"❌ 无法加载任何 DPLM 模型: {e3}")
                print("使用占位符模型进行演示...")
                
                # 如果加载失败，使用占位符模型
                class DummyModel(nn.Module):
                    def __init__(self, vocab_size=21, d_model=768, seq_len=256):
                        super().__init__()
                        self.config = type('C', (), {'hidden_size': d_model})
                        self.embed = nn.Embedding(vocab_size, d_model)
                        self.transformer = nn.TransformerEncoder(
                            nn.TransformerEncoderLayer(d_model, nhead=8, dim_feedforward=d_model*4, batch_first=True),
                            num_layers=12
                        )
                        self.head = nn.Linear(d_model, vocab_size)

                    def forward(self, input_ids, return_hidden=False, **kwargs):
                        # input_ids: (B, L)
                        x = self.embed(input_ids)
                        h = self.transformer(x)
                        logits = self.head(h)
                        if return_hidden:
                            # return logits and a pooled sequence embedding (mean pool)
                            seq_emb = h.mean(dim=1)
                            return logits, seq_emb
                        return logits

                model = DummyModel().to(device)
                print("✅ 使用占位符模型，训练可以正常进行")

    # 预假设：你已有若干 frozen modality encoders（backbone/pocket/ligand/function/text）
    # 如果没有，请在你的项目中定义并加载预训练 encoders；下面用占位器演示
    function_modalities = ['function_go', 'function_ipr', 'function_ec']
    base_modalities = ['backbone', 'ligand']
    extra_modalities = ['text']
    modality_names = base_modalities + function_modalities + extra_modalities
    modality_alias_map = {
        'function_go': 'go_terms',
        'function_ipr': 'ipr',
        'function_ec': 'ec',
        'text': 'text',
    }
    function_label_map = {
        'function_go': 'go',
        'function_ipr': 'ipr',
        'function_ec': 'ec',
        'text': 'text',
    }

    # --------
    # 2. Frozen structural encoder (optional) and MMCP components
    # --------
    d_cond = 256
    K_shared = args.K_shared
    K_per_modality = args.K_per_modality
    mmcp_indicators = {m: ModalityIndicatorHead(d_cond).to(device) for m in modality_names}
    # modality-specific queries as nn.ParameterDict to be optimized
    modality_queries = nn.ParameterDict({m: nn.Parameter(torch.randn(1, K_per_modality, d_cond)*0.02) for m in modality_names})
    mmcp_lq = MMCP_LQ(d_cond=d_cond, K_shared=K_shared, num_layers=2, n_heads=8).to(device)
    # projection to dplm model dim
    dplm_dim = model.config.hidden_size
    W_to_dplm = nn.Linear(d_cond, dplm_dim, bias=True).to(device)

    # --- 多模态编码器 ---
    enable_backbone_encoder = not getattr(args, 'disable_backbone_encoder', False)
    enable_ligand_encoder = not getattr(args, 'disable_ligand_encoder', False)
    enable_function_encoder = not getattr(args, 'disable_function_encoder', False)

    backbone_encoder = None
    if enable_backbone_encoder:
        try:
            backbone_encoder = GVPTransformerEncoder(
                freeze=args.freeze_backbone_encoder,
                output_dim=d_cond,
                use_esm=True
            ).to(device)
            if args.freeze_backbone_encoder:
                backbone_encoder.eval()
            print("✅ 已启用多模态 GVP-Transformer 结构编码器")
        except Exception as e:
            print(f"⚠️ 无法初始化GVP-Transformer编码器: {e}")
            backbone_encoder = None

    ligand_encoder = None
    if enable_ligand_encoder:
        try:
            ligand_encoder = SchNetEncoder(
                atom_embed_dim=args.ligand_atom_embed_dim,
                hidden_dim=args.ligand_hidden_dim,
                output_dim=args.ligand_output_dim,
                num_filters=args.ligand_num_filters,
                num_interactions=args.ligand_num_interactions,
                cutoff=args.ligand_cutoff
            ).to(device)
            if args.freeze_ligand_encoder:
                for p in ligand_encoder.parameters():
                    p.requires_grad = False
            print("✅ 已启用SchNet配体编码器")
        except Exception as e:
            print(f"⚠️ 无法初始化SchNet编码器: {e}")
            ligand_encoder = None

    protbert_encoder = None
    if enable_function_encoder:
        try:
            protbert_encoder = ProtBERTEncoder(
                model_name=args.protbert_model_name,
                output_dim=args.protbert_hidden_dim,
                freeze=args.freeze_protbert,
                max_length=args.protbert_max_length
            ).to(device)
            if args.freeze_protbert:
                protbert_encoder.eval()
            print("✅ 已启用ProtBERT功能注释编码器")
        except Exception as e:
            print(f"⚠️ 无法初始化ProtBERT编码器: {e}")
            protbert_encoder = None

    # create adapters and register to mid/late layers
    all_layers = find_transformer_layer_modules(model)
    if len(all_layers) == 0:
        raise RuntimeError("无法自动定位 transformer 层，请手动指定 adapter_layers。")
    num_layers = len(all_layers)
    # choose three positions: middle, 3/4, last
    idxs = sorted(set([max(0, num_layers//2 - 1), max(0, (3*num_layers)//4 - 1), max(0, num_layers - 1)]))
    adapter_layers = [all_layers[i] for i in idxs]
    adapters = attach_adapters_to_modules(adapter_layers, d_model=dplm_dim, n_heads=args.adapter_n_heads)

    # buffer for S used by hooks
    S_buf = {'S': None}
    def get_condition_fn():
        return S_buf['S']
    hook_handles = register_adapter_hooks(adapter_layers, adapters, get_condition_fn)

    # small projection g for alignment (sequence embedding -> structural embedding dim)
    align_dim = 256
    g_proj = nn.Sequential(
        nn.LayerNorm(dplm_dim),
        nn.Linear(dplm_dim, align_dim),
        nn.ReLU(),
        nn.Linear(align_dim, align_dim)
    ).to(device)
    # structure projection to align_dim
    structure_proj = nn.Linear(d_cond, align_dim).to(device)

    # --------
    # 3. Projectors for each modality (trainable)
    # --------
    # If you already have small projection heads that map encoder outputs -> d_cond, use them.
    # Here we define default linear projectors (replace with your real W_proj_m).
    default_feature_dim = getattr(args, 'structure_dim', 128)
    ligand_feature_dim = args.ligand_output_dim if ligand_encoder is not None else args.ligand_raw_dim
    function_feature_dim = args.protbert_hidden_dim if protbert_encoder is not None else args.function_feature_dim

    input_dims = {
        'backbone': d_cond if backbone_encoder is not None else default_feature_dim,
        'ligand': ligand_feature_dim,
    }
    for fm in function_modalities + ['text']:
        input_dims[fm] = function_feature_dim

    W_proj = {}
    for m in modality_names:
        in_dim = input_dims.get(m, default_feature_dim)
        if in_dim == d_cond:
            W_proj[m] = nn.Identity().to(device)
        else:
            W_proj[m] = nn.Linear(in_dim, d_cond).to(device)

    miss_token_len = max(1, getattr(args, 'miss_token_len', 1))
    miss_embeddings = nn.ParameterDict({
        m: nn.Parameter(torch.randn(1, miss_token_len, d_cond, device=device) * 0.02)
        for m in modality_names
    })

    # --------
    # 4. Optimizer: collect only desired trainable params
    # --------
    trainable_params = []
    # modality projectors
    for m in modality_names:
        trainable_params += list(W_proj[m].parameters())
    # indicators
    for m in modality_names:
        trainable_params += list(mmcp_indicators[m].parameters())
    # modality queries
    for p in modality_queries.values():
        trainable_params.append(p)
    # mmcp_lq
    trainable_params += list(mmcp_lq.parameters())
    # adapters
    for a in adapters:
        trainable_params += list(a.parameters())
    # W_to_dplm, g_proj, structure_proj
    trainable_params += list(W_to_dplm.parameters())
    trainable_params += list(g_proj.parameters())
    trainable_params += list(structure_proj.parameters())
    trainable_params += list(miss_embeddings.parameters())
    if backbone_encoder is not None and not args.freeze_backbone_encoder:
        trainable_params += list(backbone_encoder.parameters())
    if ligand_encoder is not None and not args.freeze_ligand_encoder:
        trainable_params += list(ligand_encoder.parameters())
    if protbert_encoder is not None and not args.freeze_protbert:
        trainable_params += list(protbert_encoder.parameters())

    # 使用更保守的学习率和优化器设置
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-5, eps=1e-8)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type=='cuda'))

    # --------
    # 5. DataLoader (支持Inhouse数据集)
    # --------
    if args.use_inhouse and InhouseDataset is not None:
        print("🏗️ 使用Inhouse数据集进行训练...")
        print(f"  - 数据路径: {args.inhouse_data_path}")
        print(f"  - 最大序列长度: {args.max_seq_len}")
        print(f"  - 结构特征维度: {args.structure_dim}")
        print(f"  - 只使用骨架原子: {args.use_backbone_only}")
        
            # 创建Inhouse数据集
        dataset = InhouseDataset(
            data_path=args.inhouse_data_path,
            max_seq_len=args.max_seq_len,
            structure_dim=args.structure_dim,
            use_backbone_only=args.use_backbone_only
        )
        
        # 更新投影器输入维度（仅骨架模态）
        if 'backbone' in W_proj and not isinstance(W_proj['backbone'], nn.Identity):
            W_proj['backbone'] = nn.Linear(args.structure_dim, d_cond).to(device)
        
        # 自定义collate函数处理变长序列
        def inhouse_collate_fn(batch):
            """Inhouse数据集的自定义collate函数"""
            # 找到批次中的最大序列长度
            max_len = max(item['input_ids'].size(0) for item in batch)
            
            # 填充所有序列到相同长度
            input_ids = []
            labels = []
            modalities = {'backbone': []}
            structures = []
            sequences = []
            names = []
            inhouses = []
            
            for item in batch:
                seq_len = item['input_ids'].size(0)
                
                # 填充input_ids和labels
                if seq_len < max_len:
                    pad_len = max_len - seq_len
                    input_ids.append(torch.cat([item['input_ids'], torch.zeros(pad_len, dtype=torch.long)]))
                    labels.append(torch.cat([item['labels'], torch.zeros(pad_len, dtype=torch.long)]))
                else:
                    input_ids.append(item['input_ids'])
                    labels.append(item['labels'])
                
                # 填充结构特征
                structure = item['structure']
                if structure.size(0) < max_len:
                    pad_len = max_len - structure.size(0)
                    padded_structure = torch.cat([
                        structure, 
                        torch.zeros(pad_len, structure.size(1))
                    ])
                else:
                    padded_structure = structure[:max_len]
                
                modalities['backbone'].append(padded_structure)
                structures.append(padded_structure)
                sequences.append(item['sequence'])
                names.append(item['name'])
                inhouses.append(item['inhouse'])
            
            return {
                'input_ids': torch.stack(input_ids),
                'labels': torch.stack(labels),
                'modalities': {k: torch.stack(v) for k, v in modalities.items()},
                'structure': torch.stack(structures),
                'sequence': sequences,
                'name': names,
                'inhouse': inhouses
            }
        
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=0, 
            pin_memory=(device.type=='cuda'), 
            collate_fn=inhouse_collate_fn
        )
        
        print(f"✅ Inhouse数据集加载完成: {len(dataset)} 个样本")
        
    else:
        print("📊 使用占位符数据集进行训练...")
        # Placeholder dataset: user must replace with real dataset yielding:
        # batch = {'input_ids': (B,L), 'labels': (B,L), 'modalities': {m: raw_mod_input}, 'structure': optional}
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, n=1000, seq_len=128, vocab=21):
                self.n = n
                self.seq_len = seq_len
                self.vocab = vocab
            def __len__(self):
                return self.n
            def __getitem__(self, idx):
                return {
                    'input_ids': torch.randint(0, self.vocab, (self.seq_len,)),
                    'labels': torch.randint(0, self.vocab, (self.seq_len,)),
                    # 模态占位符
                    'modalities': {
                        'backbone': torch.randn(16, args.structure_dim),
                        'ligand': torch.randn(12, args.ligand_raw_dim),
                        'function_go': torch.randn(1, args.function_feature_dim),
                        'function_ipr': torch.randn(1, args.function_feature_dim),
                        'function_ec': torch.randn(1, args.function_feature_dim),
                        'text': torch.randn(1, args.function_feature_dim),
                    },
                    # optional structure placeholder - use empty tensor instead of None
                    'structure': torch.empty(0)  # empty tensor instead of None
                }
        dataset = DummyDataset()
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=(device.type=='cuda'), collate_fn=None)

    # --------
    # Helper functions
    # --------
    def _to_device(value):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {k: _to_device(v) for k, v in value.items()}
        return value

    def encode_backbone_mod(raw):
        if raw is None:
            return None
        if isinstance(raw, dict):
            coords = raw.get('coords') or raw.get('positions') or raw.get('xyz')
            mask = raw.get('mask')
            if coords is None:
                return None
            coords = _to_device(coords)
            if mask is not None:
                mask = _to_device(mask)
            if backbone_encoder is not None:
                return backbone_encoder(coords, mask)
            return coords
        if isinstance(raw, torch.Tensor):
            return raw.to(device)
        return None

    def encode_ligand_mod(raw):
        if raw is None:
            return None
        if isinstance(raw, dict):
            atom_types = raw.get('atom_types')
            positions = raw.get('positions')
            mask = raw.get('mask')
            if atom_types is None or positions is None:
                return None
            atom_types = _to_device(atom_types).long()
            positions = _to_device(positions)
            mask = _to_device(mask) if mask is not None else None
            if ligand_encoder is not None:
                return ligand_encoder(atom_types, positions, mask)
            return positions
        if isinstance(raw, torch.Tensor):
            return raw.to(device)
        return None

    def encode_function_mod(raw, label):
        if raw is None:
            return None
        annotations = raw
        if isinstance(raw, dict):
            annotations = raw.get('text') or raw.get('annotations') or raw.get('tokens')
        if annotations is None:
            return None
        if protbert_encoder is not None:
            features = protbert_encoder(annotations, annotation_type=label)
        elif isinstance(annotations, torch.Tensor):
            features = annotations.to(device)
        else:
            return None
        if features.dim() == 1:
            features = features.unsqueeze(0)
        return features

    def encode_modality_tokens(name, raw):
        if raw is None:
            return None
        if name == 'backbone':
            return encode_backbone_mod(raw)
        if name == 'ligand':
            return encode_ligand_mod(raw)
        if name in function_modalities or name == 'text':
            label = function_label_map.get(name, 'text')
            return encode_function_mod(raw, label)
        if isinstance(raw, torch.Tensor):
            return raw.to(device)
        if isinstance(raw, dict):
            return _to_device(raw.get('features'))
        return None

    modality_dropout_prob = max(0.0, min(1.0, getattr(args, 'modality_dropout_prob', 0.0)))

    # --------
    # 6. Training loop
    # --------
    model.train()
    for epoch in range(args.epochs):
        for step, batch in enumerate(dataloader):
            # move inputs to device where needed
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # --- 1) construct condition token stream C ---
            tilde_list = []
            encoded_cache = {}
            modality_available = {}
            batch_modalities = batch.get('modalities', {}) or {}
            batch_size = input_ids.size(0)
            for m in modality_names:
                raw = batch_modalities.get(m)
                if raw is None:
                    alias = modality_alias_map.get(m)
                    if alias is not None:
                        raw = batch_modalities.get(alias, batch.get(alias))
                if raw is None:
                    raw = batch.get(m)
                encoded = encode_modality_tokens(m, raw)
                encoded_cache[m] = encoded
                drop_flag = False
                if encoded is not None and modality_dropout_prob > 0:
                    drop_flag = torch.rand(1).item() < modality_dropout_prob
                modality_available[m] = encoded is not None and not drop_flag
                if encoded is None or drop_flag:
                    tokens = miss_embeddings[m].expand(batch_size, -1, -1)
                else:
                    tokens = encoded
                    if tokens.dim() == 2:
                        tokens = tokens.unsqueeze(1)
                Zm = W_proj[m](tokens)
                Zm_tilde = mmcp_indicators[m](Zm)
                tilde_list.append(Zm_tilde)

            if len(tilde_list) == 0:
                C = torch.zeros(input_ids.size(0), 0, d_cond, device=device)
            else:
                C = torch.cat(tilde_list, dim=1)  # (B, N, d_cond)

            modality_queries_pass = {m: modality_queries[m] for m in modality_names}

            # --- 2) MMCP-LQ fusion ---
            Q_out = mmcp_lq(C, modality_queries=modality_queries_pass)  # (B, K_total, d_cond)
            S = W_to_dplm(Q_out)  # (B, K, dplm_dim)

            # set S for hooks to pick up
            S_buf['S'] = S

            # --- 3) DPLM forward (hooks will apply adapters) ---
            # it is assumed model.forward returns (logits, seq_embedding) if return_hidden=True
            try:
                outputs = model.forward(input_ids, return_hidden=True)
                if isinstance(outputs, tuple) and len(outputs) == 2:
                    logits, seq_emb = outputs
                else:
                    # 如果模型返回的不是期望的格式，尝试处理
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    elif hasattr(outputs, 'last_hidden_state'):
                        # 如果没有 logits，使用 last_hidden_state 作为 logits
                        logits = outputs.last_hidden_state
                    else:
                        logits = outputs
                    
                    batch_size = input_ids.size(0)
                    if hasattr(outputs, 'last_hidden_state'):
                        seq_emb = outputs.last_hidden_state.mean(dim=1)
                    else:
                        seq_emb = torch.randn(batch_size, 768).to(device)
            except TypeError:
                # fallback: model.forward returns logits only; we'll pool last hidden (if possible)
                outputs = model.forward(input_ids)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                elif hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif hasattr(outputs, 'last_hidden_state'):
                    logits = outputs.last_hidden_state
                else:
                    logits = outputs
                
                # try to get hidden via attribute or embedding
                if hasattr(outputs, 'last_hidden_state'):
                    seq_emb = outputs.last_hidden_state.mean(dim=1)
                elif hasattr(model, 'last_hidden_state'):
                    seq_emb = model.last_hidden_state.mean(dim=1)
                else:
                    # As last resort, mean of logits' pre-softmax embedding dimension (LESS IDEAL)
                    seq_emb = logits.detach().float().mean(dim=1)
            
            # 检查logits和seq_emb是否为NaN
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"⚠️ 检测到NaN/Inf logits，跳过此步骤")
                continue
            
            if torch.isnan(seq_emb).any() or torch.isinf(seq_emb).any():
                print(f"⚠️ 检测到NaN/Inf seq_emb，跳过此步骤")
                continue

            # --- 4) compute losses ---
            L_CE = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

            # 检查损失是否为NaN
            if torch.isnan(L_CE) or torch.isinf(L_CE):
                print(f"⚠️ 检测到NaN/Inf损失，跳过此步骤")
                continue

            # Check if structure data is available (not empty tensor)
            structure_data = batch.get('structure', None)
            if structure_data is not None and structure_data.numel() > 0:
                backbone_tokens = None
                if modality_available.get('backbone', False):
                    backbone_tokens = encoded_cache.get('backbone')
                if backbone_tokens is None:
                    backbone_tokens = structure_data.to(device)
                    if backbone_tokens.dim() == 2:
                        backbone_tokens = backbone_tokens.unsqueeze(1)
                    if backbone_tokens.size(-1) != d_cond:
                        backbone_tokens = W_proj['backbone'](backbone_tokens)
                else:
                    if backbone_tokens.dim() == 2:
                        backbone_tokens = backbone_tokens.unsqueeze(1)
                    if backbone_tokens.size(-1) != d_cond:
                        backbone_tokens = W_proj['backbone'](backbone_tokens)
                z_tokens = structure_proj(backbone_tokens)
                z_str = z_tokens.mean(dim=1)
                p = g_proj(seq_emb)
                L_align = info_nce_loss(p, z_str, temperature=0.07)
                if torch.isnan(L_align) or torch.isinf(L_align):
                    print(f"⚠️ 检测到NaN/Inf对齐损失，设置为0")
                    L_align = 0.0
            else:
                L_align = 0.0

            L_total = L_CE + args.gamma * L_align
            
            # 检查总损失
            if torch.isnan(L_total) or torch.isinf(L_total):
                print(f"⚠️ 检测到NaN/Inf总损失，跳过此步骤")
                continue

            # --- 5) backward & optimize (AMP safe) ---
            optimizer.zero_grad()
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    L_total.backward()
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()
            else:
                L_total.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

            # clear S buffer
            S_buf['S'] = None

            # logging
            if step % args.log_every == 0:
                print(f"Epoch {epoch} Step {step} L_CE={L_CE.item():.4f} L_align={float(L_align):.4f} Total={L_total.item():.4f}")

            # checkpointing, validation etc. can be added here

    # end training loop

    # cleanup hooks
    for h in hook_handles:
        try:
            h.remove()
        except Exception:
            pass

    # save final model or mmcp params
    torch.save({
        'mmcp_lq_state': mmcp_lq.state_dict(),
        'mmcp_indicators': {m: mmcp_indicators[m].state_dict() for m in mmcp_indicators},
        'W_to_dplm': W_to_dplm.state_dict(),
        'adapters': [a.state_dict() for a in adapters],
        'miss_embeddings': miss_embeddings.state_dict(),
        'backbone_encoder': backbone_encoder.state_dict() if backbone_encoder is not None else None,
        'ligand_encoder': ligand_encoder.state_dict() if ligand_encoder is not None else None,
        'protbert_encoder': protbert_encoder.state_dict() if protbert_encoder is not None else None,
    }, args.save_path or "mmcp_gen_checkpoint.pt")

    print("Training finished and checkpoint saved.")


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMCP-GEN integrated fine-tuning (template)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--K_shared", type=int, default=16)
    parser.add_argument("--K_per_modality", type=int, default=2)
    parser.add_argument("--adapter_n_heads", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_path", type=str, default="mmcp_gen_checkpoint.pt")
    parser.add_argument("--disable_backbone_encoder", action="store_true", help="禁用GVP-Transformer骨架编码器")
    parser.add_argument("--freeze_backbone_encoder", action="store_true", help="冻结骨架编码器参数")
    parser.add_argument("--disable_ligand_encoder", action="store_true", help="禁用SchNet配体编码器")
    parser.add_argument("--freeze_ligand_encoder", action="store_true", help="冻结SchNet参数")
    parser.add_argument("--disable_function_encoder", action="store_true", help="禁用ProtBERT功能编码器")
    parser.add_argument("--freeze_protbert", action="store_true", help="冻结ProtBERT参数")
    parser.add_argument("--ligand_atom_embed_dim", type=int, default=64, help="SchNet原子嵌入维度")
    parser.add_argument("--ligand_hidden_dim", type=int, default=256, help="SchNet隐藏维度")
    parser.add_argument("--ligand_output_dim", type=int, default=256, help="SchNet输出特征维度")
    parser.add_argument("--ligand_num_filters", type=int, default=64, help="SchNet滤波器数量")
    parser.add_argument("--ligand_num_interactions", type=int, default=3, help="SchNet交互层数")
    parser.add_argument("--ligand_cutoff", type=float, default=5.0, help="SchNet截断距离")
    parser.add_argument("--ligand_raw_dim", type=int, default=128, help="未使用SchNet时配体特征维度")
    parser.add_argument("--protbert_model_name", type=str, default="Rostlab/prot_bert", help="ProtBERT模型名称")
    parser.add_argument("--protbert_hidden_dim", type=int, default=256, help="ProtBERT输出特征维度")
    parser.add_argument("--protbert_max_length", type=int, default=512, help="ProtBERT最大输入长度")
    parser.add_argument("--function_feature_dim", type=int, default=128, help="功能模态占位特征维度")
    parser.add_argument("--modality_dropout_prob", type=float, default=0.1, help="模态dropout概率")
    parser.add_argument("--miss_token_len", type=int, default=1, help="缺失模态占位token长度")
    # GVP-Transformer 开关与权重
    # parser.add_argument("--use_gvp_transformer", action="store_true", help="使用GVP-Transformer作为冻结结构编码器")
    # parser.add_argument("--gvp_transformer_ckpt", type=str, default=None, help="GVP-Transformer权重路径")
    
    # Inhouse数据集相关参数
    parser.add_argument("--use_inhouse", action="store_true", help="使用Inhouse数据集进行训练")
    parser.add_argument("--inhouse_data_path", type=str, 
                       default=None,
                       help="Inhouse数据集路径")
    parser.add_argument("--max_seq_len", type=int, default=256, help="最大序列长度")
    parser.add_argument("--structure_dim", type=int, default=128, help="结构特征维度")
    parser.add_argument("--use_backbone_only", action="store_true", default=True, 
                       help="只使用骨架原子（CA）作为结构特征")
    
    args = parser.parse_args()
    main(args)
