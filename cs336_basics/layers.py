import torch
from torch import Tensor

def linear(weights: Tensor, x: Tensor) -> Tensor:
    """weights: (d_out, d_in), x: (..., d_in) -> (..., d_out)"""
    return x @ weights.transpose(0,1)

def embedding(weights: Tensor, token_ids: Tensor) -> Tensor:
    """weights: (vocab, d_model), token_ids: (...) -> (..., d_model)"""
    return weights[token_ids]

def rmsnorm(weights: Tensor, x: Tensor, eps: float) -> Tensor:
    """weights: (d_model,), x: (..., d_model)"""
    in_dtype = x.dtype
    x32 = x.to(torch.float32)
    mean_sq = (x32 * x32).mean(dim = -1, keepdim=True)
    inv_rms = torch.rsqrt(mean_sq + eps)
    y32 = x32 * inv_rms * weights
    return y32.to(in_dtype)




