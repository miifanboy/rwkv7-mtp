"""
mtp -- Variable-length Multi-Token Prediction (MTP) heads for a frozen RWKV-7.

This is the package form of the former monolithic ``mtp.py``.  The public API is
unchanged, so ``import mtp`` / ``from mtp import build_frozen_model`` keep
working; the implementation is split by concern:

======================  =====================================================
module                  contents
======================  =====================================================
:mod:`mtp.env`          ``RWKV_*`` env, CUDA/MSVC build fixes, PL shim
:mod:`mtp.tensors`      ``_shift_left`` and other pure tensor helpers
:mod:`mtp.heads`        MTP head architectures (``mlp``, ``rwkv7``)
:mod:`mtp.losses`       chunked CE, distillation and LK losses
:mod:`mtp.backbone`     frozen backbone + chained heads (``RWKVWithMTP``)
:mod:`mtp.checkpoint`   ``.pth`` loading, ``mtp_heads.pt`` save/load/inspect
:mod:`mtp.cli`          ``python -m mtp shift|inspect``
======================  =====================================================

Retrofit design
---------------
* The RWKV-7 backbone is frozen. Only the MTP heads are trained.
* K MTP heads are plugged on top of the backbone's final hidden state.
* The frozen backbone output projection (``model.head``) is shared by every
  MTP depth.  No per-depth output heads are added (a per-depth head would cost
  K * n_embd * vocab_size parameters, ~168M each at 2560x65536).

The two head architectures live in :mod:`mtp.heads`; the losses that train them
live in :mod:`mtp.losses`.
"""

from __future__ import annotations

# ``env`` MUST be imported first: it pins RWKV_JIT_ON / RWKV_HEAD_L2WRAP_CE_CHUNK
# before anything can import ``src.model``.
from . import env  # noqa: F401
from .env import _patch_cpp_extension_flags, _prepare_build_env
from .tensors import _shift_left
from .heads import MTP_ARCHS, MTPModule, RWKV7MTPModule, build_head
from .losses import (
    CE_CHUNK,
    LK_EPS,
    _ce,
    _distill_ce,
    _FusedDistillCE,
    _FusedHeadCE,
    _FusedLK,
    _head_ce,
    _lk_cuda_ext,
    _lk_loss,
    _teacher_logits,
    _teacher_topk,
    mtp_loss,
)
from .backbone import _load_rwkv_base, _make_rwkv_with_mtp, _rwkv_base_module
from .checkpoint import (
    _payload_arch,
    _resize_params_to_checkpoint,
    build_frozen_model,
    count_mtp_params,
    load_mtp_heads,
    load_pth_state_dict,
    make_args_from_config,
    mtp_parameters,
    peek_mtp_arch,
    peek_mtp_meta,
    save_mtp_heads,
)
from .cli import run_shift_selftest
from . import backbone, checkpoint, heads, losses, tensors  # noqa: F401

__all__ = [
    "MTP_ARCHS", "MTPModule", "RWKV7MTPModule", "build_head",
    "build_frozen_model", "mtp_parameters", "count_mtp_params",
    "save_mtp_heads", "load_mtp_heads", "peek_mtp_arch", "peek_mtp_meta",
    "load_pth_state_dict", "make_args_from_config",
    "mtp_loss", "_shift_left", "run_shift_selftest",
    "_prepare_build_env", "_patch_cpp_extension_flags",
    "CE_CHUNK", "LK_EPS",
    "_ce", "_head_ce", "_FusedHeadCE",
    "_teacher_topk", "_distill_ce", "_FusedDistillCE",
    "_lk_loss", "_FusedLK", "_lk_cuda_ext", "_teacher_logits",
]


def __getattr__(name: str):
    # PEP 562: ``from mtp import RWKVWithMTP`` triggers the lazy build (which
    # imports src.model and compiles the CUDA kernels).
    if name == "RWKVWithMTP":
        return _make_rwkv_with_mtp()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
