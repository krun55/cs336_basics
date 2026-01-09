import torch
import torch.nn as nn
from torch import Tensor
from .ffn import swiglu
from .attention import mha, mha_with_rope
from .rope import RoPECache
from .layers import rmsnorm

# 你可以复用 functional/layers/attention/ffn/rope 里的实现
# 这里先搭模块树，让 load_state_dict(weights) 能吃进去
class LinearWeights(nn.Module):
    def __init__(self, out_features, in_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

    def forward(self, x :Tensor) -> Tensor:
        return x @ self.weight.t()

class Attn(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta):
        super().__init__()
        self.q_proj = LinearWeights(d_model, d_model)
        self.k_proj = LinearWeights(d_model, d_model)
        self.v_proj = LinearWeights(d_model, d_model)
        self.output_proj = LinearWeights(d_model, d_model)
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
    
    def forward(self, x : Tensor) -> Tensor:
        cache = RoPECache(
        d_k=self.d_model,
        base=self.theta,
        max_seq_len=self.max_seq_len,
        device=x.device,
        dtype=x.dtype,
    )
        return mha_with_rope(self.q_proj.weight, self.k_proj.weight, self.v_proj.weight, self.output_proj.weight, x, self.num_heads, cache)

class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = LinearWeights(d_ff, d_model)
        self.w2 = LinearWeights(d_model, d_ff)
        self.w3 = LinearWeights(d_ff, d_model)
    
    def  forward(self, x : Tensor) -> Tensor:
        return swiglu(self.w1.weight, self.w2.weight, self.w3.weight, x)
    

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        
        self.attn = Attn(d_model, num_heads, max_seq_len, theta)  
        self.ln1 = LinearWeights(d_model, d_model)
        self.ffn = FFN(d_model, d_ff)
        self.ln2 = LinearWeights(d_model, d_model)
        self.max_seq_len = max_seq_len
        self.theta = theta
        # TODO: rope cache

    def forward(self, x: Tensor) -> Tensor:
        x_norm1 = rmsnorm(self.ln1.weight, x)
        x = self.Attn(x_norm1)
        x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float):
        super().__init__()
        self.token_embeddings = nn.Parameter(torch.empty(vocab_size, d_model))
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta) for _ in range(num_layers)])
        self.ln_final = nn.Parameter(torch.ones(d_model))
        self.lm_head = nn.Parameter(torch.empty(vocab_size, d_model))

    def forward(self, idx: Tensor) -> Tensor:
        raise NotImplementedError
