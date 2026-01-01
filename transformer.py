import torch
import torch.nn as nn
from torch import Tensor

# 你可以复用 functional/layers/attention/ffn/rope 里的实现
# 这里先搭模块树，让 load_state_dict(weights) 能吃进去

class Attn(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.q_proj = nn.Parameter(torch.empty(d_model, d_model))  # 只是占位示例：你最后会换成你自己的 Linear/Parameter 结构
        self.k_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.v_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.output_proj = nn.Parameter(torch.empty(d_model, d_model))

class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(d_model, d_ff))
        self.w2 = nn.Parameter(torch.empty(d_ff, d_model))
        self.w3 = nn.Parameter(torch.empty(d_model, d_ff))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        self.attn = nn.Module()  # TODO: 换成你自己的 Attention 模块，但名字必须叫 attn，且有 q_proj/k_proj/v_proj/output_proj
        self.ln1 = nn.Parameter(torch.ones(d_model))
        self.ffn = nn.Module()   # TODO: 有 w1/w2/w3
        self.ln2 = nn.Parameter(torch.ones(d_model))
        # TODO: rope cache

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float):
        super().__init__()
        self.token_embeddings = nn.Parameter(torch.empty(vocab_size, d_model))
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta) for _ in range(num_layers)])
        self.ln_final = nn.Parameter(torch.ones(d_model))
        self.lm_head = nn.Parameter(torch.empty(vocab_size, d_model))

    def forward(self, idx: Tensor) -> Tensor:
        raise NotImplementedError
