from torch import Tensor
from .functional import silu

def swiglu(w1: Tensor, w2: Tensor, w3: Tensor, x: Tensor) -> Tensor:
    """w1:(d_ff,d_model), w2:(d_model,d_ff), w3:(d_ff,d_model), x:(...,d_model)->(...,d_model)"""
    a = silu(w1(x))
    b = w3(x)
    g = a * b
    return w3(g)

