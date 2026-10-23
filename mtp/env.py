"""
Process/build environment for the MTP retrofit.

This module owns everything that has to happen *before* the heavy
``src.model`` import:

* sets the ``RWKV_*`` environment variables ``src.model`` reads at import time;
* puts the project root on ``sys.path`` so ``import src.model`` works from any
  launch directory;
* locates the CUDA toolkit / MSVC preprocessor and patches
  ``torch.utils.cpp_extension.load`` so the vendor ``.cu`` sources build under
  CUDA 13;
* installs a tiny ``pytorch_lightning`` shim when the real package is broken.

It is imported first by :mod:`mtp` and is cheap (no torch import at module
level beyond the type used by the shim).
"""

from __future__ import annotations

import os
import sys
import types

# Project root = parent of the ``mtp`` package (this file is ``mtp/env.py``).
# Everything (``src/``, ``cuda/``, ``cuda_compat.h``) is resolved against it so
# scripts can live in subdirectories (``scripts/``) and still work.
_THIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


# ---------------------------------------------------------------------------
# 0. Environment for src.model (must run before importing it)
# ---------------------------------------------------------------------------
# src.model defines a module-level helper named ``_forward_op`` five times (once
# per kernel).  With RWKV_JIT_ON=0 the later definitions overwrite the earlier
# ones and every wrapper calls the wrong function, so JIT must stay ON (which is
# what train.py uses anyway).
if os.environ.get("RWKV_JIT_ON", "1") != "1":
    print("[mtp] forcing RWKV_JIT_ON=1: src/model.py's JIT_ON=0 path has a "
          "duplicate `_forward_op` definition bug")
os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_MY_TESTING"] = os.environ.get("RWKV_MY_TESTING", "x070")
os.environ.setdefault("RWKV_KERNEL", "")
os.environ["RWKV_HEAD_SIZE"] = os.environ.get("RWKV_HEAD_SIZE", "64")
if int(os.environ.get("RWKV_HEAD_L2WRAP_CE_CHUNK", "0")) != 0:
    raise RuntimeError(
        "RWKV_HEAD_L2WRAP_CE_CHUNK must be 0 for MTP: the fused head+CE path "
        "never materializes per-depth logits."
    )
os.environ["RWKV_HEAD_L2WRAP_CE_CHUNK"] = "0"


# ---------------------------------------------------------------------------
# 5. Build environment + pytorch_lightning shim
# ---------------------------------------------------------------------------

def _prepare_build_env() -> None:
    """Best-effort setup so the RWKV CUDA kernels can compile.

    * MSVC needs the conforming preprocessor for CUDA 13's CCCL headers.
    * ``CUDA_HOME`` must point at a CUDA toolkit with ``nvcc``.
    * ``-std=c++20`` (added in the load() patch below) avoids a torch 2.10 /
      MSVC ``C2872: 'std': ambiguous symbol`` failure in nvcc's host pass.
    """
    import torch

    if os.name == "nt":
        cl = os.environ.get("CL", "")
        if "/Zc:preprocessor" not in cl:
            os.environ["CL"] = (cl + " /Zc:preprocessor").strip()

    if not os.environ.get("CUDA_HOME"):
        import shutil
        nvcc = shutil.which("nvcc")
        if nvcc:
            os.environ["CUDA_HOME"] = os.path.dirname(os.path.dirname(os.path.abspath(nvcc)))
        else:
            base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
            if os.path.isdir(base):
                versions = sorted(os.listdir(base))
                if versions:
                    os.environ["CUDA_HOME"] = os.path.join(base, versions[-1])

    if not os.environ.get("TORCH_CUDA_ARCH_LIST") and torch.cuda.is_available():
        caps = sorted({f"{torch.cuda.get_device_capability(i)[0]}."
                       f"{torch.cuda.get_device_capability(i)[1]}"
                       for i in range(torch.cuda.device_count())})
        os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(caps)


def _patch_cpp_extension_flags() -> None:
    """Make space-containing CUDA flags work with modern torch/ninja.

    ``src/model.py`` passes ``extra_cuda_cflags=[..., "-Xptxas -O3", ...]`` as a
    single list element.  Newer torch/ninja forward each element verbatim to
    nvcc, which then rejects the whole string as one unknown option.  Splitting
    such elements into separate argv entries fixes it without touching
    model.py.
    """
    try:
        from torch.utils import cpp_extension
    except Exception:
        return
    if getattr(cpp_extension, "_mtp_flag_patch", False):
        return

    _orig_load = cpp_extension.load

    _compat = os.path.join(_THIS_DIR, "cuda_compat.h").replace("\\", "/")

    def _split(flags):
        if not flags:
            return flags
        out = []
        for f in flags:
            if isinstance(f, str) and " " in f:
                out.extend(f.split())
            else:
                out.append(f)
        return out

    def load(*args, **kwargs):
        for key in ("extra_cuda_cflags", "extra_cflags", "extra_ldflags"):
            if key in kwargs:
                kwargs[key] = _split(kwargs[key])
        # CUDA 13 dropped atomicAdd(float2*, float2); restore it via a
        # force-included compat header without editing the vendor .cu files.
        cc = list(kwargs.get("extra_cuda_cflags") or [])
        if "-include" not in cc:
            cc = ["-include", _compat] + cc
        if "-std=c++20" not in cc:
            cc = cc + ["-std=c++20"]
        kwargs["extra_cuda_cflags"] = cc
        return _orig_load(*args, **kwargs)

    cpp_extension.load = load
    cpp_extension._mtp_flag_patch = True


def _install_lightning_shim() -> None:
    if "pytorch_lightning" in sys.modules:
        return
    try:
        import pytorch_lightning  # noqa: F401
        return
    except Exception as e:  # pragma: no cover - environment dependent
        print(f"[mtp] pytorch_lightning unavailable ({type(e).__name__}: {e}); using shim")

    for m in list(sys.modules):
        if (m == "pytorch_lightning" or m.startswith("pytorch_lightning.")
                or m == "torchmetrics" or m.startswith("torchmetrics.")):
            del sys.modules[m]

    def rank_zero_info(*a, **k):
        if a:
            print(*a)

    def rank_zero_only(fn):
        return fn

    pl = types.ModuleType("pytorch_lightning")
    pl.__version__ = "0.0.0-shim"
    import torch.nn as nn
    pl.LightningModule = nn.Module

    utilities = types.ModuleType("pytorch_lightning.utilities")
    utilities.rank_zero_info = rank_zero_info
    utilities.rank_zero_only = rank_zero_only
    pl.utilities = utilities

    strategies = types.ModuleType("pytorch_lightning.strategies")

    class DeepSpeedStrategy:  # placeholder, never instantiated here
        pass

    strategies.DeepSpeedStrategy = DeepSpeedStrategy
    pl.strategies = strategies

    sys.modules["pytorch_lightning"] = pl
    sys.modules["pytorch_lightning.utilities"] = utilities
    sys.modules["pytorch_lightning.strategies"] = strategies
