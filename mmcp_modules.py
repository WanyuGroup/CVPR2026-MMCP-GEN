# src/mmcp_modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Callable, Tuple, Optional

# -------------------------
# Modality Indicator Head
# -------------------------
class ModalityIndicatorHead(nn.Module):
    """
    Prepend a single learnable modality indicator token to a modality token sequence.
    Input: tokens (B, L, d_cond)
    Output: (B, L+1, d_cond) with indicator prepended
    """
    def __init__(self, d_cond: int):
        super().__init__()
        # initialize small random vector; will be learned.
        self.indicator = nn.Parameter(torch.randn(1, d_cond) * 0.02)

    def forward(self, tokens: torch.Tensor):
        # tokens: (B, L, d)
        B = tokens.shape[0]
        ind = self.indicator.unsqueeze(0).expand(B, 1, -1)  # (B,1,d)
        return torch.cat([ind, tokens], dim=1)


# -------------------------
# Cross-Attention Adapter
# -------------------------
class CrossAttentionAdapter(nn.Module):
    """
    Lightweight cross-attention adapter to inject condition summaries into DPLM hidden states.
    Query: sequence hidden states H (B, L_seq, d_model)
    Key/Value: condition summary S (B, K, d_model)
    Returns: delta to add to H (B, L_seq, d_model)
    """
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1, scale_init: float = 0.1):
        super().__init__()
        self.layernorm = nn.LayerNorm(d_model)
        # using batch_first MultiheadAttention (PyTorch 1.11+)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.out_proj = nn.Linear(d_model, d_model)
        # learnable scale for residual injection (start small)
        self.alpha = nn.Parameter(torch.tensor(scale_init, dtype=torch.float32))

    def forward(self, H: torch.Tensor, S: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        """
        H: (B, L_seq, d_model)
        S: (B, K, d_model)
        returns: (B, L_seq, d_model)
        """
        H_norm = self.layernorm(H)
        # MultiheadAttention signature: (query, key, value)
        attn_out, _ = self.cross_attn(query=H_norm, key=S, value=S, key_padding_mask=None, attn_mask=attn_mask)
        out = self.out_proj(attn_out)
        return self.alpha * out


# -------------------------
# MMCP-LQ: Learnable Query Fusion
# -------------------------
class MMCP_LQ(nn.Module):
    """
    Single-tower learnable query fusion.
    - Accepts concatenated condition tokens C: (B, N, d_cond)
    - Accepts modality-specific query dict (optional): name -> (1, K_m, d_cond)
    - Maintains shared queries internally.
    - Runs a small Transformer encoder (self-attention) jointly over tokens+queries.
    - Returns updated queries Q' of shape (B, K_total, d_cond)
    """
    def __init__(self,
                 d_cond: int,
                 K_shared: int = 8,
                 num_layers: int = 2,
                 n_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.d_cond = d_cond
        self.K_shared = K_shared
        # shared queries are trainable
        self.shared_queries = nn.Parameter(torch.randn(1, K_shared, d_cond) * 0.02)
        # Transformer encoder (batch_first)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_cond, nhead=n_heads,
                                                   dim_feedforward=d_cond * 4,
                                                   dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # small layernorm on outputs
        self.out_norm = nn.LayerNorm(d_cond)

    def forward(self, C: torch.Tensor, modality_queries: Optional[Dict[str, torch.Tensor]] = None):
        """
        C: (B, N, d_cond) - concatenated modality tokens (each has been prepended with its indicator)
        modality_queries: dict modality-> tensor (1, K_m, d_cond)  (optional)
        returns: Q_out (B, K_total, d_cond)
        """
        B = C.shape[0]
        queries = [self.shared_queries.expand(B, -1, -1)]  # (B, K_shared, d)
        if modality_queries:
            # modality_queries values expected as (1, Km, d_cond) or (Km, d_cond)
            for k, v in modality_queries.items():
                if v is None:
                    continue
                qv = v
                if qv.dim() == 2:
                    qv = qv.unsqueeze(0)
                queries.append(qv.expand(B, -1, -1))
        Q = torch.cat(queries, dim=1)  # (B, K_total, d)
        T = torch.cat([C, Q], dim=1)   # (B, N+K, d)
        T_out = self.transformer(T)    # (B, N+K, d)
        # extract last K tokens as updated queries
        K_total = Q.size(1)
        Q_out = T_out[:, -K_total:, :].contiguous()
        Q_out = self.out_norm(Q_out)
        return Q_out  # (B, K_total, d_cond)


# -------------------------
# Helper: Attach adapters to model (non-invasive)
# -------------------------
def attach_adapters_to_modules(modules: List[nn.Module], d_model: int, n_heads: int = 8) -> List[CrossAttentionAdapter]:
    """
    Given a list of nn.Module objects (transformer blocks), attach a CrossAttentionAdapter to each by setting
    attribute 'mmcp_adapter' on the module. Returns the list of created adapters (same order).
    """
    adapters = []
    for i, layer in enumerate(modules):
        adapter = CrossAttentionAdapter(d_model=d_model, n_heads=n_heads)
        setattr(layer, 'mmcp_adapter', adapter)
        adapters.append(adapter)
    return adapters


# -------------------------
# Helper: Register forward hooks calling adapter
# -------------------------
def register_adapter_hooks(layer_modules: List[nn.Module],
                           adapters: List[CrossAttentionAdapter],
                           get_condition_fn: Callable[[], Optional[torch.Tensor]]) -> List[torch.utils.hooks.RemovableHandle]:
    """
    Register forward hooks on each layer module to apply the adapter. The get_condition_fn must return
    the current batch S (B, K, d_model) or None.
    Returns list of handles to remove later.
    """
    handles = []

    for layer, adapter in zip(layer_modules, adapters):
        # create a hook capturing the adapter
        def make_hook(adapter):
            def hook(module, input, output):
                # Many transformer layers return a tensor H; if tuple, take first
                H = output[0] if isinstance(output, (tuple, list)) else output
                S = get_condition_fn()
                if S is None:
                    return output
                try:
                    delta = adapter(H, S)  # (B, L, d)
                    H_new = H + delta
                    # replace first element if tuple
                    if isinstance(output, (tuple, list)):
                        out_list = list(output)
                        out_list[0] = H_new
                        return tuple(out_list)
                    else:
                        return H_new
                except Exception as e:
                    # On any error, fallback to original output (do not crash)
                    return output
            return hook
        h = layer.register_forward_hook(make_hook(adapter))
        handles.append(h)
    return handles
