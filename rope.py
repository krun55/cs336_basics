import torch
from torch import Tensor

class RoPECache:
    def __init__(self, d_k: int, theta: float, max_seq_len: int, device=None, dtype=None):
        # TODO: build cos/sin cache (max_seq_len, d_k/2)
        raise NotImplementedError

def apply_rope(x: Tensor, token_positions: Tensor, cache: RoPECache) -> Tensor:
    """x:(..., T, d_k), token_positions:(..., T)"""
    raise NotImplementedError
