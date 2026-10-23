"""
MTP head architectures.

Every head maps ``(backbone hidden, committed-token embedding)`` to a draft
hidden that the *frozen* backbone head then projects to logits.  Two families:

* :class:`~mtp.heads.mlp.MTPModule` -- position-local MLP fusion (stateless);
* :class:`~mtp.heads.rwkv7.RWKV7MTPModule` -- one full RWKV-7 block, recurrent
  at inference.

:data:`MTP_ARCHS` is the registry consumed by ``--mtp-arch`` and
:func:`build_head` is the single construction seam.
"""

from __future__ import annotations

from .mlp import MTPModule
from .rwkv7 import RWKV7MTPModule

# Supported MTP head architectures.  ``mlp`` is the original position-local
# module (stateless across draft steps); ``rwkv7`` is a full RWKV-7 block that
# mixes across stream positions and therefore carries a recurrent state at
# inference (see ``RWKV7MTPModule`` and ``spec_decode.RWKV7HeadRunner``).
MTP_ARCHS = ("mlp", "rwkv7")


def build_head(arch: str, args, eagle3_fusion: bool = True,
               num_depths: int = 1):
    """Construct one MTP head module for ``arch`` (an :data:`MTP_ARCHS` name).

    ``eagle3_fusion`` / ``num_depths`` only apply to the ``rwkv7`` head (EAGLE-3
    multi-layer fusion + per-depth input LayerNorms under a shared head); the
    ``mlp`` head ignores them.
    """
    if arch == "mlp":
        return MTPModule(args.n_embd)
    if arch == "rwkv7":
        return RWKV7MTPModule(args, eagle3_fusion=eagle3_fusion,
                              num_depths=num_depths)
    raise ValueError(f"unknown --mtp-arch {arch!r}; choose from {MTP_ARCHS}")


__all__ = ["MTPModule", "RWKV7MTPModule", "MTP_ARCHS", "build_head"]
