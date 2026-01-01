import math
import torch
from torch import Tensor

def silu(x: Tensor) -> Tensor:
    raise NotImplementedError

def softmax(x: Tensor, dim: int) -> Tensor:
    raise NotImplementedError

def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """inputs: (B, V), targets: (B,) -> scalar"""
    raise NotImplementedError

def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None) -> Tensor:
    """Q: (..., Q, d_k), K: (..., K, d_k), V: (..., K, d_v), mask: (..., Q, K) bool"""
    raise NotImplementedError

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    raise NotImplementedError
