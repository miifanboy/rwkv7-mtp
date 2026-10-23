"""
Generic logits-in MTP loss (main head + all depths), used by the CPU tests.

The training path does not use this: it goes through ``_head_ce`` so the
``(B, T, V)`` logits are never materialised.  This helper exists for callers
that already have logits and want the same weighted-CE objective.
"""

from __future__ import annotations

from typing import Dict, List

import torch

from ..tensors import _shift_left
from .common import _ce


def mtp_loss(
    main_logits: torch.Tensor,
    mtp_logits: List[torch.Tensor],
    targets: torch.Tensor,
    decay: float = 0.8,
    ignore_index: int = 0,
    return_parts: bool = False,
):
    """Weighted CE over the main head and all MTP depths.

    ``weight_k = decay ** (k + 1)`` for MTP depth ``k`` (0-indexed); the main
    head has weight 1.  The total is divided by the sum of weights so its scale
    is comparable to a plain CE.
    """
    parts: Dict[str, torch.Tensor] = {}
    ce_main = _ce(main_logits, targets, ignore_index)
    loss = ce_main
    total_w = 1.0
    parts["main"] = ce_main.detach()

    for k, logits in enumerate(mtp_logits):
        tgt_k = _shift_left(targets, k + 1, pad=ignore_index)
        ce_k = _ce(logits, tgt_k, ignore_index)
        w = decay ** (k + 1)
        loss = loss + w * ce_k
        total_w += w
        parts[f"mtp_{k + 1}"] = ce_k.detach()

    loss = loss / total_w
    parts["total"] = loss.detach()
    if return_parts:
        return loss, parts
    return loss
