from torch import Tensor
from .functional import silu
from .layers import linear

def swiglu(w1: Tensor, w2: Tensor, w3: Tensor, x: Tensor) -> Tensor:
    """w1:(d_ff,d_model), w2:(d_model,d_ff), w3:(d_ff,d_model), x:(...,d_model)->(...,d_model)"""
    gate = silu(linear(w1, x))
    value = linear(w3, x)
    hidden = gate * value
    out = linear(w2, hidden)
    return out

