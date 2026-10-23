"""
Losses for the frozen-backbone MTP retrofit.

Two families:

* hard-CE based -- :func:`_head_ce` (fused head + CE) and :func:`_distill_ce`
  (hard-CE + top-k self-distillation KL);
* LK / acceptance based -- :func:`_lk_loss` (hybrid KL+TV or ``-log alpha``),
  which directly optimises the speculative-decoding acceptance rate.

All of them share the chunked recompute-in-backward discipline
(:mod:`mtp.losses.common`) so no ``(B, T, V)`` tensor is ever retained.
"""

from .common import CE_CHUNK, _ce
from .ce import _FusedHeadCE, _head_ce
from .distill import _teacher_topk, _FusedDistillCE, _distill_ce
from .lk import (
    LK_EPS,
    _FusedLK,
    _lk_cuda_ext,
    _lk_loss,
    _teacher_logits,
)
from .mtp import mtp_loss
from . import common, ce, distill, lk, mtp  # noqa: F401  (expose submodules)

__all__ = [
    "CE_CHUNK", "_ce",
    "_FusedHeadCE", "_head_ce",
    "_teacher_topk", "_FusedDistillCE", "_distill_ce",
    "LK_EPS", "_FusedLK", "_lk_cuda_ext", "_lk_loss", "_teacher_logits",
    "mtp_loss",
]
