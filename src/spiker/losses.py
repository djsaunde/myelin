"""Training losses for spiking workloads."""

from __future__ import annotations

import torch
from torch import nn


def spike_rate_loss(spikes: torch.Tensor, target_rate: float | torch.Tensor = 0.1) -> torch.Tensor:
    """Mean-squared error between observed and target spike rate."""

    target = torch.as_tensor(target_rate, dtype=spikes.dtype, device=spikes.device)
    return (spikes.mean() - target).square()


class SpikeRateLoss(nn.Module):
    """Module wrapper around ``spike_rate_loss``."""

    def __init__(self, target_rate: float = 0.1) -> None:
        super().__init__()
        self.target_rate = target_rate

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        return spike_rate_loss(spikes, self.target_rate)
