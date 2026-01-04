import torch
from torch import Tensor
from .functional import scaled_dot_product_attention
from .rope import RoPECache, apply_rope

def mha(
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    x: Tensor,
    num_heads: int,
) -> Tensor:
    """x:(..., T, d_model) -> (..., T, d_model)"""
    raise NotImplementedError

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
    raise NotImplementedError
