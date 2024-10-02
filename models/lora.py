import math
import inspect
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from collab_utils.collaboration_strategies import CollaborationStrategy

# Overwriting the methods of nn.Linear:
# https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
class LoRALinear(nn.Linear):

    def __init__(self,
                 # nn.Linear parameters
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device=None,
                 dtype=None,
                 # LoRA parameters
                 lora_rank: int = 0,
                 lora_alpha: float = 0.0,
                 lora_dropout: float = 0.0,
                ) -> None:
        nn.Linear.__init__(
            self,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype
        )

        # LoRA stuff
        self.has_weights_merged = False
        if lora_rank > 0:
            self.lora_dropout = nn.Dropout(lora_dropout)

            self.lora_scaling = lora_alpha / np.sqrt(lora_rank)
            self.lora_A = nn.Parameter(torch.empty((lora_rank, self.in_features), device=device, dtype=dtype))
            self.lora_B = nn.Parameter(torch.empty((self.out_features, lora_rank), device=device, dtype=dtype))

            self.lora_A.requires_grad = False
            self.lora_B.requires_grad = False

            self.reset_parameters()

    def is_lora(self) -> bool:
        return hasattr(self, 'lora_A')

    def reset_parameters(self) -> None:
        nn.Linear.reset_parameters(self)
        if self.is_lora():
            torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5)) # Same as nn.Linear
            torch.nn.init.zeros_(self.lora_B)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = nn.Linear.forward(self, input)
        if not self.has_weights_merged and self.is_lora():
            # h = Wx + BAx * scaling
            x += self.lora_scaling * F.linear(
                F.linear(
                    self.lora_dropout(input),
                    self.lora_A
                ),
                self.lora_B
            )
        return x

    def extra_repr(self) -> str:
        out = nn.Linear.extra_repr(self)
        if self.is_lora():
            out += f', lora_rank={self.lora_A.shape[0]}, lora_scaling={self.lora_scaling}, lora_dropout={self.lora_dropout.p}'
        return out

    def train(self, mode: bool = True) -> "LoRALinear":
        nn.Linear.train(self, mode)
        if self.has_weights_merged and self.is_lora():
            # de-merge weights, i.e., remove BA from W = W + BA
            self.weight.data -= self.lora_scaling * self.lora_B @ self.lora_A
            self.has_weights_merged = False
        return self

    def eval(self) -> "LoRALinear":
        nn.Linear.eval(self)
        if not self.has_weights_merged and self.is_lora():
            # merge weights, i.e., add BA to W
            self.weight.data += self.lora_scaling * self.lora_B @ self.lora_A
            self.has_weights_merged = True
        return self

def get_ft_model(model: nn.Module, collaboration_strategy: CollaborationStrategy) -> nn.Module:
    for name, param in model.named_parameters():
        if collaboration_strategy.is_optimized(name):
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model