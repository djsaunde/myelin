"""Surrogate spike functions shared by modules and custom autograd."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch

SurrogateFn = Callable[[torch.Tensor], torch.Tensor]
SurrogateName = Literal[
    "sigmoid",
    "fast_sigmoid",
    "atan",
    "triangular",
    "superspike",
    "multi_gaussian",
]
SURROGATE_NAMES: tuple[SurrogateName, ...] = (
    "sigmoid",
    "fast_sigmoid",
    "atan",
    "triangular",
    "superspike",
    "multi_gaussian",
)


def sigmoid_surrogate(centered: torch.Tensor) -> torch.Tensor:
    """Smooth sigmoid surrogate."""

    return torch.sigmoid(centered)


def fast_sigmoid_surrogate(centered: torch.Tensor) -> torch.Tensor:
    """Cheaper smooth surrogate with sigmoid-like range."""

    return 0.5 * (centered / (1.0 + centered.abs()) + 1.0)


def atan_surrogate(centered: torch.Tensor) -> torch.Tensor:
    """Smooth arctangent surrogate."""

    return torch.atan(torch.pi * centered / 2.0) / torch.pi + 0.5


def triangular_surrogate(centered: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear triangular surrogate."""

    return (1.0 - centered.abs()).clamp(min=0.0, max=1.0)


def superspike_surrogate(centered: torch.Tensor) -> torch.Tensor:
    """SuperSpike primitive whose derivative is ``1 / (1 + |x|)^2``.

    This is intended mainly for hard-forward surrogate gradients. Unlike the
    normalized fast-sigmoid primitive, the smooth forward primitive is not
    clamped to ``[0, 1]``.
    """

    return centered / (1.0 + centered.abs()) + 0.5


def _normal_cdf(centered: torch.Tensor, *, mean: float, sigma: float) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf((centered - mean) / (1.4142135623730951 * sigma)))


def multi_gaussian_surrogate(centered: torch.Tensor) -> torch.Tensor:
    """Smooth primitive for a normalized three-component Gaussian-mixture derivative."""

    return (
        0.6 * _normal_cdf(centered, mean=0.0, sigma=0.5)
        + 0.2 * _normal_cdf(centered, mean=1.0, sigma=1.0)
        + 0.2 * _normal_cdf(centered, mean=-1.0, sigma=1.0)
    )


def surrogate_derivative(centered: torch.Tensor, name: SurrogateName) -> torch.Tensor:
    """Derivative of a built-in surrogate with respect to ``centered``."""

    if name == "sigmoid":
        value = sigmoid_surrogate(centered)
        return value * (1.0 - value)
    if name == "fast_sigmoid":
        return 0.5 / (1.0 + centered.abs()).square()
    if name == "atan":
        scaled = torch.pi * centered / 2.0
        return 0.5 / (1.0 + scaled.square())
    if name == "triangular":
        return torch.where(
            centered.abs() < 1.0,
            -centered.sign(),
            torch.zeros_like(centered),
        )
    if name == "superspike":
        return 1.0 / (1.0 + centered.abs()).square()
    if name == "multi_gaussian":
        inv_sqrt_2pi = 0.3989422804014327
        center = 0.6 * (inv_sqrt_2pi / 0.5) * torch.exp(-0.5 * (centered / 0.5).square())
        right = 0.2 * inv_sqrt_2pi * torch.exp(-0.5 * (centered - 1.0).square())
        left = 0.2 * inv_sqrt_2pi * torch.exp(-0.5 * (centered + 1.0).square())
        return center + right + left


def hard_surrogate_spike(centered: torch.Tensor, surrogate: SurrogateFn) -> torch.Tensor:
    """Binary forward spike with gradients supplied by a smooth surrogate."""

    hard_spike = (centered >= 0).to(centered.dtype)
    smooth_spike = surrogate(centered)
    return hard_spike + smooth_spike - smooth_spike.detach()


def surrogate_from_name(name: SurrogateName) -> SurrogateFn:
    """Return a surrogate function by stable API name."""

    if name == "sigmoid":
        return sigmoid_surrogate
    if name == "fast_sigmoid":
        return fast_sigmoid_surrogate
    if name == "atan":
        return atan_surrogate
    if name == "triangular":
        return triangular_surrogate
    if name == "superspike":
        return superspike_surrogate
    if name == "multi_gaussian":
        return multi_gaussian_surrogate


def surrogate_name(surrogate: SurrogateFn) -> SurrogateName:
    """Return the stable API name for a built-in surrogate function."""

    if surrogate is sigmoid_surrogate:
        return "sigmoid"
    if surrogate is fast_sigmoid_surrogate:
        return "fast_sigmoid"
    if surrogate is atan_surrogate:
        return "atan"
    if surrogate is triangular_surrogate:
        return "triangular"
    if surrogate is superspike_surrogate:
        return "superspike"
    if surrogate is multi_gaussian_surrogate:
        return "multi_gaussian"

    msg = (
        "Triton surrogate backward supports only built-in surrogate functions: "
        "sigmoid_surrogate, fast_sigmoid_surrogate, atan_surrogate, "
        "triangular_surrogate, superspike_surrogate, and "
        "multi_gaussian_surrogate."
    )
    raise ValueError(msg)
