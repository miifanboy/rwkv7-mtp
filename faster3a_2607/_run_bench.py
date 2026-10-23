# Runs rwkv7_fast_v3a.py's CLI after applying mtp's build-env fixes
# (CUDA_HOME detection, CL=/Zc:preprocessor, arch list), PLUS Windows-specific
# build fixes for the v3a extensions themselves:
#   * no -std=c++20 injection: nvcc 13.2 miscompiles/parses rwkv7_v3a_ops.cu's
#     x.data_ptr<dtype>() under -std=c++20 (plain torch -std=c++17 builds fine);
#   * explicit cublas.lib/cublasLt.lib linking: the engine calls cublasLt
#     directly, which links transitively on Linux but must be explicit on MSVC.
# Usage: python _run_bench.py --model <pth> --cases 64x1,... [--deltalog]
import os
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
sys.path.insert(0, _PARENT)
sys.path.insert(0, _HERE)

import mtp  # noqa: E402

mtp._prepare_build_env()

# Capture the un-patched load BEFORE mtp's flag patch wraps it.
from torch.utils import cpp_extension as _ce  # noqa: E402

_orig_load = _ce.load
_IS_WIN = os.name == "nt"
_CUDA_LIB = os.path.join(os.environ.get("CUDA_HOME", ""), "lib", "x64")


def _split(flags):
    out = []
    for f in flags or ():
        if isinstance(f, str) and " " in f:
            out.extend(f.split())
        else:
            out.append(f)
    return out


def load(*args, **kwargs):
    for key in ("extra_cuda_cflags", "extra_cflags", "extra_ldflags"):
        if key in kwargs:
            kwargs[key] = _split(kwargs[key])
    # Per-extension -std: nvcc 13.2 misparses rwkv7_v3a_ops.cu's x.data_ptr<dtype>()
    # under -std=c++20, but rwkv7_wkv_fp32_v2.cu hits the torch-2.10/MSVC
    # C2872 'std': ambiguous error without it. Everything else builds on c++17.
    name = args[0] if args else kwargs.get("name", "")
    cc = list(kwargs.get("extra_cuda_cflags") or [])
    if "fp32" in str(name) and "-std=c++20" not in cc:
        cc.append("-std=c++20")
    kwargs["extra_cuda_cflags"] = cc
    if _IS_WIN:
        ld = list(kwargs.get("extra_ldflags") or [])
        for lib in ("cublas.lib", "cublasLt.lib"):
            p = os.path.join(_CUDA_LIB, lib)
            if p not in ld:
                ld.append(p)
        kwargs["extra_ldflags"] = ld
    return _orig_load(*args, **kwargs)


_ce.load = load

runpy.run_path(os.path.join(_HERE, "rwkv7_fast_v3a.py"), run_name="__main__")