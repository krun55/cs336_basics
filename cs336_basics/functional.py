import math
import torch
from torch import Tensor

def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)

def softmax(x: Tensor, dim: int) -> Tensor:
    if dim < 0:
        dim += x.dim()
    m = torch.amax(x, axis=dim, keepdim = True)
    z = x - m
    e = torch.exp(z)
    s = torch.sum(e, axis=dim, keepdim = True)
    y = e / s
    return y

def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """inputs: (B, V), targets: (B,) -> scalar"""
    logits_fp32 = torch.to_float32(inputs)
    dim = -1
    m = torch.amax(logits_fp32, axis = dim, keepdim=True)
    z = logits_fp32 - m
    logsumexp = torch.log(torch.sum(torch.exp(z), axis=dim, keepdim = True))
    z_y = torch.gather(targets, axis=dim, keepdim = True)
    output = torch.mean(logsumexp-z_y)
    return output


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
