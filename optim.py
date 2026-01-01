import torch
from torch.optim import Optimizer

class AdamW(Optimizer):
    def __init__(self, params, lr: float, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        raise NotImplementedError
