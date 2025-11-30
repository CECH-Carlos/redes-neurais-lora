import torch
import torch.nn as nn
import pytorch_lightning as pl


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, scaling=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.randn(out_features, r) * 0.02)
        self.r = r
        self.scaling = scaling

    def forward(self, x):
        original = torch.matmul(x, self.weight.t())
        lora_update = x @ self.lora_A.t() @ self.lora_B.t() * self.scaling
        return original + lora_update