import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.merged = False
        self.merge_weights = merge_weights


def _summarize_cond_from_input(x: torch.Tensor) -> torch.Tensor:
    if x.dim() <= 2:
        return x
    reduce_dims = tuple(range(1, x.dim() - 1))
    return x.mean(dim=reduce_dims)


class LayerHarmonizer(nn.Module):
    """
    Small per-layer network that predicts adaptive sine coefficients.
    """

    def __init__(self, cond_dim: int, hidden: int = 64, out_dim: int = 4) -> None:
        super().__init__()
        self.lora_harmonizer = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.lora_harmonizer(cond)


class AdaSineLoRALinear(nn.Linear, LoRALayer):
    """
    Minimal Adaptive Sine LoRA linear layer for latent projections.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        merge_weights: bool = False,
        cond_dim: Optional[int] = None,
        harmonizer_hidden: int = 64,
        omega0: float = 300.0,
        init_identity: bool = False,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )
        self.omega0 = float(omega0)
        self._s = 2.0 * math.sqrt(in_features)

        if init_identity and in_features == out_features:
            with torch.no_grad():
                self.weight.zero_()
                self.weight.copy_(torch.eye(in_features, dtype=self.weight.dtype, device=self.weight.device))
                if self.bias is not None:
                    self.bias.zero_()

        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False
            if cond_dim is None:
                cond_dim = in_features
            self.lora_harmonizer = LayerHarmonizer(cond_dim, hidden=harmonizer_hidden)
        else:
            self.lora_harmonizer = None

        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        if hasattr(self, "lora_A"):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        base = F.linear(x, self.weight, bias=self.bias)
        if not (self.r > 0 and not self.merged):
            return base

        core = F.linear(self.lora_dropout(x), self.lora_A)
        core = F.linear(core, self.lora_B)

        if self.lora_harmonizer is None:
            return base + core * self.scaling

        if cond is None:
            cond = _summarize_cond_from_input(x)
        coef = self.lora_harmonizer(cond)
        a, b, c, d = coef.unbind(dim=-1)
        # Broadcast per-sample harmonizer coefficients across every non-batch
        # dimension of the LoRA core, e.g. (B, N, D) -> (B, 1, 1).
        num_new_dims = core.dim() - 1
        slicing = (slice(None),) + (None,) * num_new_dims

        amp = torch.exp(a)[slicing]
        freq = torch.exp(b)[slicing]
        phase = c[slicing]
        bias = d[slicing]
        adaptive = amp * torch.sin(freq * self.omega0 * core + phase) / self._s + bias
        return base + adaptive * self.scaling
