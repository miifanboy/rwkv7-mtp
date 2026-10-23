"""Pure tensor helpers with no backbone/CUDA dependency (unit-testable on CPU)."""

from __future__ import annotations

import torch


def _shift_left(t: torch.Tensor, shift: int, pad: int = 0) -> torch.Tensor:
    """Left-shift ``t`` by ``shift`` along the last dim, padding with ``pad``.

    ``out[..., i] = t[..., i + shift]`` and positions where ``i + shift >= T``
    are filled with ``pad``.
    """
    if shift == 0:
        return t
    if shift < 0:
        raise ValueError(f"_shift_left only supports shift >= 0, got {shift}")
    T = t.shape[-1]
    out = t.new_full(t.shape, pad)
    if shift < T:
        out[..., : T - shift] = t[..., shift:]
    return out
