import torch
from torch import Tensor
from .functional import scaled_dot_product_attention
from .rope import RoPECache, apply_rope
from layers import linear

def mha(
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    x: Tensor,
    num_heads: int,
) -> Tensor:
    """x:(..., T, d_model) -> (..., T, d_model)"""
    b, t, d = x.size()
    d_head = dk / num_heads
    Q = linear(q_proj_weight,x).view()
    K = linear(k_proj_weight,x)
    V = linear(v_proj_weight,x)
    return scaled_dot_product_attention(Q,K,V)

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
    Q = linear(q_proj_weight,x)
    K = linear(k_proj_weight,x)
    V = linear(v_proj_weight,x)
    Q = apply_rope(Q)
    K = apply_rope(K)
    return scaled_dot_product_attention(Q,K,V)
