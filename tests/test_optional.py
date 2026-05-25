from __future__ import annotations

import spiker._optional as optional


def test_optional_availability_is_cached(monkeypatch) -> None:
    def fail_find_spec(name: str):
        raise AssertionError(f"find_spec should not be called for {name}")

    monkeypatch.setattr(optional, "find_spec", fail_find_spec)

    assert optional.has_torch() is optional._TORCH_AVAILABLE  # pyright: ignore[reportPrivateUsage]
    assert optional.has_triton() is optional._TRITON_AVAILABLE  # pyright: ignore[reportPrivateUsage]
