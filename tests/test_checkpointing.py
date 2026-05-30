from __future__ import annotations

import pytest

from myelin.checkpointing import (
    parse_checkpoint_size,
    recommended_checkpoint_size,
    resolve_checkpoint_size,
)


def test_recommended_checkpoint_size_policies() -> None:
    assert recommended_checkpoint_size(100, policy="memory") == 10
    assert recommended_checkpoint_size(100, policy="balanced") == 25
    assert recommended_checkpoint_size(100, policy="speed") == 50


def test_resolve_checkpoint_size_caps_integer_at_timesteps() -> None:
    assert resolve_checkpoint_size(10, 25) == 10


def test_resolve_checkpoint_size_rejects_nonpositive_integer() -> None:
    with pytest.raises(ValueError, match="positive"):
        resolve_checkpoint_size(10, 0)


def test_parse_checkpoint_size_accepts_policy_or_integer() -> None:
    assert parse_checkpoint_size("balanced") == "balanced"
    assert parse_checkpoint_size("12") == 12


def test_parse_checkpoint_size_rejects_invalid_value() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        parse_checkpoint_size("not-a-policy")
