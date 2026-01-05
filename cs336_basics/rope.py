import torch
from torch import Tensor

class RoPECache:
    def __init__(self, d_k: int, base: float, max_seq_len: int, device=None, dtype=None):
        # TODO: build cos/sin cache (max_seq_len, d_k/2)
        assert d_k % 2 == 0
        half = d_k // 2
        device = device if device is not None else device = torch.device("cpu")
        pos = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        k = torch.arange(half, device=device, dtype=torch.float32)
        inv_freq = base ** (-2*k/d_k)
        angles = pos[:, None] * inv_freq[None, :]
        self.cos = torch.cos(angles)
        self.sin = torch.sin(angles)

def apply_rope(x: Tensor, token_positions: Tensor, cache: RoPECache) -> Tensor:
    """x:(..., T, d_k), token_positions:(..., T)"""
    assert x.shape[-1] % 2 == 0
    x_even = x[..., :, 0::2]
    x_odd = x[..., :, 1::2]
    token_positions = token_positions.to(device=x.device, dtype=torch.long)
    cos = cache.cos.to(device=x.device, dtype=x.dtype)[token_positions]
    sin = cache.sin.to(device=x.device, dtype=x.dtype)[token_positions]
    y_even = x_even * cos - x_odd * sin
    y_odd= x_even * sin + x_odd * cos
    out = torch.empty_like(x)
    out[..., :, 0::2] = y_even
    out[..., :, 1::2] = y_odd
    return out
