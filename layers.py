import torch
from torch import Tensor

def linear(weights: Tensor, x: Tensor) -> Tensor:
    """weights: (d_out, d_in), x: (..., d_in) -> (..., d_out)"""
    raise NotImplementedError

def embedding(weights: Tensor, token_ids: Tensor) -> Tensor:
    """weights: (vocab, d_model), token_ids: (...) -> (..., d_model)"""
    raise NotImplementedError

def rmsnorm(weights: Tensor, x: Tensor, eps: float) -> Tensor:
    """weights: (d_model,), x: (..., d_model)"""
    raise NotImplementedError
