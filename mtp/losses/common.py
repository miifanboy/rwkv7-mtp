"""
Shared, memory-bounded cross-entropy primitives for the MTP losses.

The obvious MTP loss

    logits = head(z)                        # (B, T, V) bf16
    loss   = F.cross_entropy(logits.float(), targets)

is what does not fit on a 12 GB card.  At T=4096 / V=65536 one depth costs:

    logits              (B, T, V) bf16   0.50 GiB   transient
    .float()            (B, T, V) fp32   1.00 GiB   transient
    saved log-softmax   (B, T, V) fp32   1.00 GiB   RETAINED to backward

and the retained row grows by ~1 GiB per MTP depth, because every depth's
``ce_k`` keeps its own graph alive.  ``del logits_k`` reclaims only the first
row -- it is the autograd graph, not the Python name, that pins the tensor.

Every loss in this package is therefore chunked over rows, so the transient
peak is ``CE_CHUNK * vocab_size * 4`` bytes instead of T-proportional.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F

# ``RWKV_MTP_CE_CHUNK`` overrides the chunk size (default 256 rows).
CE_CHUNK = max(1, int(os.environ.get("RWKV_MTP_CE_CHUNK", "256")))


def _ce(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int,
        chunk: Optional[int] = None) -> torch.Tensor:
    """Cross-entropy on upcast (fp32) logits, with fp32 accumulation.

    Same value as ``F.cross_entropy(logits.reshape(-1, V).float(),
    targets.reshape(-1), ignore_index=...)``, but the fp32 upcast is done one
    row-chunk at a time so the transient peak does not scale with T.

    Difference from the full-tensor ``mean`` reduction: when *every* target is
    ``ignore_index`` that returned NaN, whereas summing chunks returns 0.0.
    """
    chunk = CE_CHUNK if chunk is None else chunk
    flat = logits.reshape(-1, logits.size(-1))
    tgt = targets.reshape(-1)
    loss_sum = torch.zeros((), dtype=torch.float32, device=logits.device)
    n_valid = 0
    for start in range(0, flat.size(0), chunk):
        end = min(start + chunk, flat.size(0))
        loss_sum = loss_sum + F.cross_entropy(
            flat[start:end].float(), tgt[start:end],
            ignore_index=ignore_index, reduction="sum",
        )
        n_valid += int((tgt[start:end] != ignore_index).sum())
    return loss_sum / max(n_valid, 1)
