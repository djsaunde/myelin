"""Checkpoint-size policy helpers for time-recompute training paths."""

from __future__ import annotations

import math
from typing import Literal, TypeAlias

CheckpointPolicy: TypeAlias = Literal["memory", "balanced", "speed"]
CheckpointSize: TypeAlias = int | CheckpointPolicy
_CHECKPOINT_POLICIES = frozenset(("memory", "balanced", "speed"))


def recommended_checkpoint_size(
    timesteps: int,
    *,
    policy: CheckpointPolicy = "balanced",
) -> int:
    """Return a conservative checkpoint chunk size for a time length.

    ``memory`` keeps the chunk scratch small, ``speed`` reduces the number of
    chunks, and ``balanced`` sits between them. These are deterministic
    heuristics, not a substitute for measuring hot workloads.
    """

    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    if policy == "memory":
        return max(1, min(timesteps, math.ceil(math.sqrt(timesteps))))
    if policy == "balanced":
        return max(1, min(timesteps, math.ceil(timesteps / 4)))
    if policy == "speed":
        return max(1, min(timesteps, math.ceil(timesteps / 2)))
    raise ValueError("policy must be 'memory', 'balanced', or 'speed'")


def resolve_checkpoint_size(timesteps: int, checkpoint_size: CheckpointSize) -> int:
    """Resolve a concrete checkpoint size from an integer or named policy."""

    if isinstance(checkpoint_size, int):
        if checkpoint_size <= 0:
            raise ValueError("checkpoint_size must be positive")
        return min(checkpoint_size, timesteps)
    return recommended_checkpoint_size(timesteps, policy=checkpoint_size)


def parse_checkpoint_size(value: str) -> CheckpointSize:
    """Parse a positive integer or checkpoint-size policy string."""

    if value in _CHECKPOINT_POLICIES:
        return value  # pyright: ignore[reportReturnType]
    try:
        checkpoint_size = int(value)
    except ValueError as exc:
        expected = ", ".join(sorted(_CHECKPOINT_POLICIES))
        raise ValueError(
            f"checkpoint_size must be a positive integer or one of: {expected}"
        ) from exc
    if checkpoint_size <= 0:
        raise ValueError("checkpoint_size must be positive")
    return checkpoint_size
