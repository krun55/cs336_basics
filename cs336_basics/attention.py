import torch
from torch import Tensor
from .functional import scaled_dot_product_attention
from .rope import RoPECache, apply_rope
from .layers import linear

def mha(
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    x: Tensor,
    num_heads: int,
) -> Tensor:
    """x:(..., T, d_model) -> (..., T, d_model)"""
    *batch, t, d = x.size()
    assert d % num_heads == 0
    d_head = d // num_heads
    Q = linear(q_proj_weight,x).view(*batch, t, num_heads, d_head).transpose(-3,-2)
    K = linear(k_proj_weight,x).view(*batch, t, num_heads, d_head).transpose(-3,-2)
    V = linear(v_proj_weight,x).view(*batch, t, num_heads, d_head).transpose(-3,-2)
    indx = torch.arange(t, device=x.device)
    mask = indx[:, None] >= indx[None, :]
    O = scaled_dot_product_attention(Q,K,V, mask=mask)
    O = O.transpose(-3,-2).contiguous().view(*batch,t,d)
    out = linear(o_proj_weight,O)
    return out

def mha_with_rope(
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    x: Tensor,
    num_heads: int,
    rope_cache: RoPECache,
    token_positions: Tensor | None = None,
) -> Tensor:
    *batch, t, d = x.size()
    assert d % num_heads == 0
    d_head = d // num_heads
    if token_positions is None:
        # Default to [0..T-1], broadcast across batch dims
        base_pos = torch.arange(t, device=x.device, dtype=torch.long)
        if len(batch) > 0:
            token_positions = base_pos.reshape((1,) * len(batch) + (t,)).expand(*batch, t)
        else:
            token_positions = base_pos
    Q = linear(q_proj_weight,x).view(*batch, t, num_heads, d_head).transpose(-3,-2)
    K = linear(k_proj_weight,x).view(*batch, t, num_heads, d_head).transpose(-3,-2)
    V = linear(v_proj_weight,x).view(*batch, t, num_heads, d_head).transpose(-3,-2)
    indx = torch.arange(t, device=x.device)
    mask = indx[:, None] >= indx[None, :]
    Q = apply_rope(Q,token_positions,rope_cache)
    K = apply_rope(K,token_positions,rope_cache)
    O = scaled_dot_product_attention(Q,K,V,mask=mask)
    O = O.transpose(-3,-2).contiguous().view(*batch,t,d)
    out = linear(o_proj_weight,O)
    return out
