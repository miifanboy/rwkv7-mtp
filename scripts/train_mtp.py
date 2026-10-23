"""
train_mtp.py -- train only the MTP heads on a frozen RWKV-7 backbone.

Example
-------
    python scripts/train_mtp.py \
        --data data/train.bin --data-dtype uint16 \
        --pth rwkv7-g1j-2.9b-20260831-ctx16384.pth \
        --config config.json \
        --num-mtp 3 --ctx-len 4096 --batch-size 1 --grad-acc 8 \
        --lr 2e-4 --steps 5000 --decay 0.8 \
        --out mtp_out

The backbone is never updated; only ``model.mtp_heads.*`` is optimized.  The
output is ``<out>/mtp_heads.pt`` (MTP weights + metadata), reloadable with
``mtp.load_mtp_heads``.

This is a plain PyTorch loop on purpose: it gives direct control over which
parameters are optimized and avoids the PL/DeepSpeed plumbing in train.py.
"""

from __future__ import annotations

# --- project-root bootstrap --------------------------------------------------
# Allow `python scripts/<name>.py` / `python tests/<name>.py` from anywhere:
# put the project root (for `import mtp`, `import src.model`) and the scripts/
# directory (for sibling tool imports) on sys.path.
import os as _os
import sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, _os.path.join(_ROOT, "scripts")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# -----------------------------------------------------------------------------

import argparse
import datetime as _dt
import json
import math
import os
import random
import sys
import time
import numpy as np
import torch

import mtp
from mtp import build_frozen_model, mtp_parameters, save_mtp_heads
from trackio_helper import TrackioRun


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

_DTYPES = {"uint16": np.uint16, "uint32": np.uint32, "int32": np.int32}


def _emit_metrics(out_dir: str, rec: dict) -> None:
    """Append one record to ``<out>/metrics.jsonl`` (one JSON object per line).

    Machine-readable sibling of the human-readable ``train_log.txt``:
    ``parse_train_log.py`` rebuilds this file from the text log (backfill for
    runs started before this existed), ``plot_metrics.py`` renders it.
    """
    def _clean(v):
        # NaN/Inf would produce non-standard JSON tokens; null instead.
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v

    rec = {k: _clean(v) for k, v in rec.items()}
    path = os.path.join(out_dir, "metrics.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


class BinDataset:
    """Flat token array (``.bin``) accessed via memmap; yields (x, y) chunks.

    ``start``/``stop`` restrict sampling to a half-open token range, which lets
    one file serve as both train and validation via an in-file split (see
    ``--val-fraction``) with no overlap and no duplicate on disk.
    """

    def __init__(self, path: str, ctx_len: int, dtype: str = "uint16",
                 start: int = 0, stop: int | None = None):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Training data not found: {path}\n"
                f"Pass --data <path-to-flat-token-.bin>. Expected a headerless flat "
                f"array of token ids (dtype {dtype})."
            )
        if dtype not in _DTYPES:
            raise ValueError(f"Unsupported --data-dtype {dtype!r}; choose from {list(_DTYPES)}")
        self.path = path
        self.ctx_len = ctx_len
        self.data = np.memmap(path, dtype=_DTYPES[dtype], mode="r")
        n_total = len(self.data)
        self.start = int(start)
        self.stop = n_total if stop is None else int(stop)
        if not (0 <= self.start < self.stop <= n_total):
            raise ValueError(
                f"bad token range [{self.start}, {self.stop}) for {path} ({n_total} tokens)"
            )
        self.n_tokens = self.stop - self.start
        self.max_start = self.n_tokens - (ctx_len + 1)
        if self.max_start <= 0:
            raise ValueError(
                f"{path}[{self.start}:{self.stop}] has {self.n_tokens} tokens, "
                f"not enough for ctx_len={ctx_len}"
            )

    def sample(self, rng: random.Random, start: int | None = None):
        if start is None:
            start = rng.randint(0, self.max_start)
        base = self.start + start
        chunk = np.asarray(self.data[base:base + self.ctx_len + 1]).astype(np.int64)
        x = torch.from_numpy(chunk[:-1].copy())
        y = torch.from_numpy(chunk[1:].copy())
        return x, y


def count_tokens(path: str, dtype: str) -> int:
    """Token count of a headerless flat .bin (without mapping it)."""
    return os.path.getsize(path) // np.dtype(_DTYPES[dtype]).itemsize


def val_cut(path: str, dtype: str, n_total: int, val_fraction: float,
            search: int = 8192) -> int:
    """Index of the first document boundary at/after the hold-out point.

    ``generate_data.py`` appends a single EOS (token id 0) after every
    conversation, so snapping the cut to just past a 0 keeps train and val from
    sharing a half-document.  Falls back to the raw cut if no EOS is nearby.
    """
    cut = int(n_total * (1.0 - val_fraction))
    cut = max(1, min(cut, n_total - 1))
    data = np.memmap(path, dtype=_DTYPES[dtype], mode="r")
    seg = np.asarray(data[cut:min(n_total, cut + search)])
    hits = np.flatnonzero(seg == 0)
    if hits.size:
        return cut + int(hits[0]) + 1
    return cut


def _all_offsets(n_tokens: int, ctx_len: int, max_chunks: int) -> list[int]:
    max_start = max(1, n_tokens - (ctx_len + 1))
    if max_chunks <= 1:
        return [0]
    step = max(1, max_start // max_chunks)
    return list(range(0, max_start, step))[:max_chunks]


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def cosine_lr(step: int, total: int, base: float, final: float, warmup: int) -> float:
    if warmup > 0 and step < warmup:
        return base * (0.01 + 0.99 * step / warmup)
    if total <= warmup:
        return final
    progress = min(1.0, max(0.0, (step - warmup) / (total - warmup)))
    return final + 0.5 * (base - final) * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train MTP heads on a frozen RWKV-7")
    ap.add_argument("--data", default="data/train.bin")
    ap.add_argument("--data-dtype", default="uint16", choices=list(_DTYPES))
    ap.add_argument("--val-data", default="", help="optional held-out .bin for val loss")
    ap.add_argument("--val-fraction", type=float, default=0.0,
                    help="hold out this fraction of --data (tail, snapped to a "
                         "document boundary) as validation; ignored if --val-data is set")
    ap.add_argument("--pth", default="rwkv7-g1j-2.9b-20260831-ctx16384.pth")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--load-mtp", default="", help="resume from an mtp_heads.pt")

    ap.add_argument("--num-mtp", type=int, default=3)
    ap.add_argument("--shared-head", action="store_true",
                    help="FastMTP style: ONE MTPModule applied --num-mtp times "
                         "recursively during training (matches the recursive "
                         "draft loop used in speculative decoding).  Checkpoint-"
                         "compatible with a non-shared --num-mtp 1 model.")
    ap.add_argument("--mtp-arch", default=None, choices=list(mtp.MTP_ARCHS),
                    help="MTP head architecture: 'mlp' (position-local, the "
                         "original) or 'rwkv7' (one full RWKV-7 block; carries "
                         "a recurrent state at inference and needs extra VRAM "
                         "for the WKV backward state).  Default: auto-detect "
                         "from --load-mtp when resuming, else 'mlp'.")
    ap.add_argument("--eagle3-fusion", dest="eagle3_fusion", action="store_true",
                    default=None,
                    help="EAGLE-3 multi-layer feature fusion (rwkv7 head only): "
                         "tap early/middle/final backbone layers, fuse them "
                         "[0, 0, I]-initialized into the depth-0 state input.  "
                         "Default: on for fresh rwkv7 runs, from the checkpoint "
                         "when resuming.")
    ap.add_argument("--no-eagle3-fusion", dest="eagle3_fusion",
                    action="store_false",
                    help="disable EAGLE-3 fusion (single-layer depth-0 input, "
                         "the pre-EAGLE-3 behavior)")
    ap.add_argument("--ctx-len", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="sequences per micro-step (true batching; VRAM scales "
                         "~linearly with this).  Effective batch = "
                         "batch-size x grad-acc sequences.  The RWKV-7 training "
                         "kernels are batch-aware, so >1 works.")
    ap.add_argument("--grad-acc", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr-final", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--decay", type=float, default=0.8)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--ignore-index", type=int, default=0)
    ap.add_argument("--skip-main-ce", action="store_true",
                    help="skip the frozen main-head CE (~3.5%% of step time; it has "
                         "zero gradient and is only an in-distribution monitor)")
    ap.add_argument("--distill-topk", type=int, default=0,
                    help="if > 0, add top-k self-distillation KL to the MTP depth "
                         "losses: match the frozen backbone's own top-k distribution "
                         "at the aligned position (0 = off).  Memory-safe: same "
                         "chunked recomputation scheme as the fused CE.")
    ap.add_argument("--distill-lambda", type=float, default=0.5,
                    help="weight of the KL term relative to the hard-CE anchor "
                         "(only used when --distill-topk > 0)")
    ap.add_argument("--kl-only", action="store_true",
                    help="train the MTP depths on the top-k KL term ONLY, "
                         "excluding the hard-CE anchor (requires --distill-topk "
                         "> 0 and --distill-lambda > 0).  Depth loss becomes "
                         "distill_lambda * KL(top-k) against the frozen "
                         "backbone's own top-k distribution.  The main-head CE "
                         "is also dropped from the training total (it has zero "
                         "gradient anyway) but stays in the val report as an "
                         "in-distribution monitor.")
    ap.add_argument("--mtp-loss", default="ce", choices=["ce", "lk-hybrid", "lk-alpha"],
                    help="MTP depth loss: 'ce' = hard-CE (+ top-k self-distillation "
                         "KL when --distill-topk > 0).  'lk-hybrid' / 'lk-alpha' = "
                         "LK losses (docs/lk-loss.md): directly optimize the "
                         "speculative-decoding acceptance rate "
                         "alpha = sum min(p, q) = 1 - TV(p, q) against the frozen "
                         "backbone teacher over the FULL vocabulary (replaces the "
                         "whole depth loss; --distill-topk / --kl-only must be off). "
                         "'lk-hybrid' = adaptive lam*KL + (1-lam)*TV with "
                         "lam = exp(-eta * mean alpha); 'lk-alpha' = -log(alpha). "
                         "Memory-safe: same chunked recomputation scheme as the "
                         "fused CE.")
    ap.add_argument("--lk-eta", type=float, default=3.0,
                    help="eta of the hybrid LK lambda schedule "
                         "lam = exp(-eta * mean_alpha): higher eta switches from "
                         "KL to TV (direct acceptance-rate) optimization earlier. "
                         "Paper default 3.0 (only used with --mtp-loss lk-hybrid)")
    ap.add_argument("--lk-teacher-cache", default="auto", choices=["auto", "on", "off"],
                    help="LK mode only: build the full-vocab teacher logits table "
                         "ONCE per micro-step (~0.5 GiB bf16 at ctx 4096) and share "
                         "it across depths, instead of recomputing teacher GEMMs "
                         "per depth in forward and backward (~40%% step-time win). "
                         "auto = enable when the buffer fits in ~1 GiB")

    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--val-chunks", type=int, default=16)
    ap.add_argument("--out", default="mtp_out")
    ap.add_argument("--trackio-project", default="",
                    help="trackio project name (LOCAL logging only; view with "
                         "'trackio show --project <name>'); empty (default) "
                         "disables trackio logging entirely")
    ap.add_argument("--trackio-run", default="",
                    help="trackio run name (default: the --out directory "
                         "basename, so restarting from a checkpoint with the "
                         "same --out resumes the same run)")
    ap.add_argument("--check-only", action="store_true",
                    help="build the model, verify the freeze, print stats, exit")
    return ap.parse_args(argv)


def _resolve_device(device: str, dtype_str: str) -> torch.dtype:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available to PyTorch, but the RWKV-7 kernels require it.\n"
            "Install a CUDA-enabled torch build for your driver, e.g.\n"
            "  pip install --index-url https://download.pytorch.org/whl/cu128 torch\n"
            "and a matching torchvision if pytorch_lightning/torchmetrics need it.\n"
            f"Current torch: {torch.__version__}, cuda build: {torch.version.cuda}"
        )
    return torch.bfloat16 if dtype_str == "bfloat16" else torch.float32


@torch.no_grad()
def _validate(model, ds: BinDataset, offsets, batch_size, ignore_index, device,
              distill_topk: int = 0, distill_lambda: float = 0.5,
              kl_only: bool = False, lk_mode: str = "", lk_eta: float = 3.0,
              lk_teacher_cache: str = "auto"):
    """Average every loss component over the held-out chunks.

    Always includes the main head, whatever ``--skip-main-ce`` says.  Two
    reasons: it is only ~3.5% of a forward pass, and ``forward_loss`` normalises
    by a weight that counts the main slot either way -- so excluding it from the
    numerator alone would report a total scaled by (K+1-K)/(K+2) and make the
    number incomparable to anything.  It is also the readout that tells you
    whether the held-out text is in distribution at all.
    """
    model.eval()
    acc: dict[str, float] = {}
    n = 0
    vb = max(1, batch_size)
    # Forward-only probe: without no_grad every chunk builds (and then discards)
    # a full graph for all MTP depths.
    with torch.no_grad():
        for i in range(0, len(offsets), vb):
            group = offsets[i:i + vb]
            xs, ys = [], []
            for start in group:
                x, y = ds.sample(random.Random(0), start=start)
                xs.append(x)
                ys.append(y)
            x = torch.stack(xs).to(device)
            y = torch.stack(ys).to(device)
            _, parts = model.forward_loss(
                x, y, decay=1.0, ignore_index=ignore_index, return_parts=True,
                include_main=True,
                distill_topk=distill_topk, distill_lambda=distill_lambda,
                kl_only=kl_only, lk_mode=lk_mode, lk_eta=lk_eta,
                lk_teacher_cache=lk_teacher_cache,
            )
            # parts[...] is the token-weighted mean over this microbatch's
            # sequences.  Scale it back up by the sequence count before
            # dividing by the running sequence total, or the result is
            # deflated by ~(n_sequences / n_microbatches) = batch_size (this
            # bug made val loss report 4x low at batch_size=4).
            for k, v in parts.items():
                acc[k] = acc.get(k, 0.0) + v.item() * len(group)
            n += len(group)
    model.train()
    return {k: v / max(1, n) for k, v in acc.items()}


def main(argv=None) -> None:
    args = parse_args(argv)
    lk_mode = {"ce": "", "lk-hybrid": "hybrid", "lk-alpha": "lk_alpha"}[
        args.mtp_loss]
    if args.kl_only and (args.distill_topk <= 0 or args.distill_lambda <= 0):
        raise SystemExit(
            "[train_mtp] --kl-only requires --distill-topk > 0 and "
            "--distill-lambda > 0 (there is no KL without a teacher "
            "top-k table)")
    if lk_mode and (args.distill_topk > 0 or args.kl_only):
        raise SystemExit(
            "[train_mtp] --mtp-loss lk-* replaces the whole depth loss; "
            "--distill-topk and --kl-only must be off")
    dtype = _resolve_device(args.device, args.dtype)

    # Pin the process to the target GPU BEFORE any CUDA work.  The RWKV CUDA
    # kernels launch on the *current* device (the vendor .cu files call
    # getCurrentCUDAStream() / bare <<<>>> launches with no device guard), so
    # running with --device cuda:1 while the process current device is still
    # cuda:0 executes every custom kernel on the wrong GPU against cuda:1
    # pointers.  The corruption surfaces later as bogus cuBLAS errors
    # (CUBLAS_STATUS_ALLOC_FAILED on cublasCreate / CUBLAS_STATUS_NOT_SUPPORTED
    # on cublasGemmEx) at the first linear layer.  On cuda:0 this is invisible
    # because the current device happens to match.
    if args.device.startswith("cuda") and ":" in args.device:
        torch.cuda.set_device(args.device)
    if args.device.startswith("cuda"):
        ci = torch.cuda.current_device()
        caps = torch.cuda.get_device_capability(ci)
        print(f"[train_mtp] GPU {ci}: {torch.cuda.get_device_name(ci)} "
              f"(sm_{caps[0]}{caps[1]}, "
              f"{torch.cuda.get_device_properties(ci).total_memory / 2**20:.0f} MiB)")

    # Resolve the head arch: explicit flag wins; otherwise auto-detect from the
    # resumed checkpoint; otherwise 'mlp'.
    if args.mtp_arch is None:
        args.mtp_arch = mtp.peek_mtp_arch(args.load_mtp) if args.load_mtp else "mlp"
    if args.load_mtp:
        ckpt_arch = mtp.peek_mtp_arch(args.load_mtp)
        if ckpt_arch != args.mtp_arch:
            raise SystemExit(
                f"[train_mtp] --load-mtp {args.load_mtp} was trained with "
                f"--mtp-arch {ckpt_arch} but --mtp-arch {args.mtp_arch} was "
                f"selected; pass --mtp-arch {ckpt_arch} (or drop the flag)."
            )

    # Resolve EAGLE-3 fusion: explicit flag wins; otherwise the resumed
    # checkpoint's setting; otherwise on for fresh rwkv7 runs, off otherwise.
    ckpt_eagle3 = (mtp.peek_mtp_meta(args.load_mtp).get("eagle3_fusion", False)
                   if args.load_mtp else None)
    if args.eagle3_fusion is None:
        args.eagle3_fusion = bool(ckpt_eagle3) if args.load_mtp \
            else (args.mtp_arch == "rwkv7")
    elif ckpt_eagle3 is not None and bool(ckpt_eagle3) != args.eagle3_fusion:
        raise SystemExit(
            f"[train_mtp] --load-mtp {args.load_mtp} was trained with "
            f"eagle3_fusion={ckpt_eagle3} but --eagle3-fusion="
            f"{args.eagle3_fusion} was selected; pass the matching "
            f"(--no-)eagle3-fusion flag (or drop it)."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "train_log.txt")
    log_f = open(log_path, "a", encoding="utf-8")
    # 1-based index of this NEW RUN header inside train_log.txt.  parse_train_log.py
    # numbers the headers the same way, and metrics.jsonl records carry run_idx so
    # different eras of the same out-dir (resume/restart) never collide.
    with open(log_path, encoding="utf-8") as _f:
        run_idx = sum(1 for ln in _f if ln.startswith("NEW RUN")) + 1
    log_f.write(f"NEW RUN {_dt.datetime.now().isoformat()} {vars(args)}\n")
    log_f.flush()
    run_name = os.path.basename(os.path.normpath(args.out))
    run_start_rec = {"t": _dt.datetime.now().isoformat(timespec="seconds"),
                     "type": "run_start", "src": "live", "run": run_name,
                     "run_idx": run_idx, "args": vars(args)}
    _emit_metrics(args.out, run_start_rec)

    print(f"[train_mtp] torch={torch.__version__} cuda={torch.version.cuda} device={args.device}")

    print("[train_mtp] building frozen model (this compiles CUDA kernels on first run)...")
    model, margs = build_frozen_model(
        args.pth, args.config, num_mtp=args.num_mtp, ctx_len=args.ctx_len,
        dtype=dtype, device=args.device, shared_head=args.shared_head,
        arch=args.mtp_arch, eagle3_fusion=args.eagle3_fusion,
        warm_start_heads=not args.load_mtp,
    )
    print(f"[train_mtp] mtp arch: {args.mtp_arch}")
    if args.mtp_arch == "rwkv7":
        print(f"[train_mtp] eagle3 fusion: {args.eagle3_fusion} "
              f"(depth-0 state input = fused early/middle/final backbone taps)"
              if args.eagle3_fusion else
              "[train_mtp] eagle3 fusion: off (single-layer depth-0 input)")
    if args.shared_head:
        print(f"[train_mtp] shared-head mode: 1 MTPModule applied recursively "
              f"K={args.num_mtp} times (FastMTP style)")
    if args.mtp_arch == "rwkv7":
        print("[train_mtp] NOTE: the rwkv7 head keeps a WKV backward state per "
              "depth (~0.5-2 GB extra at ctx 4096 depending on batch); reduce "
              "--batch-size or --ctx-len if OOM.")
    if args.ignore_index != getattr(margs, "pad_token_id", args.ignore_index):
        print(f"[train_mtp] WARNING: --ignore-index={args.ignore_index} != "
              f"config pad_token_id={getattr(margs, 'pad_token_id', None)}")
    if args.ctx_len > margs.ctx_len:
        raise ValueError(f"ctx_len {args.ctx_len} > model ctx_len {margs.ctx_len}")
    if args.ctx_len % 16 != 0:
        raise ValueError("ctx_len must be a multiple of 16 (CUDA kernel CHUNK_LEN)")

    # -- resume ---------------------------------------------------------------
    # mtp_heads.pt carries the weights + meta (step, tokens_seen).  Optimizer
    # moments and the RNG states live in trainer_state.pt next to it (saved by
    # _save below), so a resumed run continues the LR schedule position, the
    # Adam moment estimates and the exact data-sampling stream instead of
    # restarting all of them from zero.
    resume = None
    start_step = 0
    start_tokens = 0
    if args.load_mtp:
        meta = mtp.load_mtp_heads(model, args.load_mtp)
        print(f"[train_mtp] resumed MTP heads from {args.load_mtp} ({meta})")
        start_step = int(meta.get("step", -1)) + 1
        start_tokens = int(meta.get("tokens_seen", 0))
        if start_step <= 0:
            print("[train_mtp] checkpoint records no step; schedule starts at 0")
            start_step = 0
        else:
            print(f"[train_mtp] continuing LR schedule from step {start_step} "
                  f"of {args.steps} ({start_tokens:,} tokens seen)")
        ts_path = os.path.join(os.path.dirname(os.path.abspath(args.load_mtp)),
                               "trainer_state.pt")
        if os.path.exists(ts_path):
            resume = torch.load(ts_path, map_location="cpu", weights_only=False)
            ts_step = int(resume.get("step", start_step - 1))
            if ts_step != start_step - 1:
                print(f"[train_mtp] WARNING: trainer_state step {ts_step} != "
                      f"checkpoint step {start_step - 1}; trusting the checkpoint step")
            print(f"[train_mtp] trainer state found at {ts_path} (step {ts_step})")
            if resume.get("torch_rng_state") is not None:
                try:
                    torch.set_rng_state(resume["torch_rng_state"])
                except Exception as e:
                    print(f"[train_mtp] WARNING: torch RNG state not restored: {e}")
            if resume.get("cuda_rng_state") is not None:
                try:
                    torch.cuda.set_rng_state(resume["cuda_rng_state"], args.device)
                except Exception as e:
                    print(f"[train_mtp] WARNING: cuda RNG state not restored: {e}")
        else:
            print(f"[train_mtp] NOTE: no trainer_state.pt beside {args.load_mtp}; "
                  f"Adam moments are NOT resumed (weights + LR schedule position are)")
        if start_step >= args.steps:
            print(f"[train_mtp] checkpoint already at step {start_step} >= --steps "
                  f"{args.steps}; nothing to train")
            return

    # ---- freeze / param report -------------------------------------------------
    params = mtp_parameters(model)
    assert params, "no trainable MTP parameters found"
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert all("mtp_heads" in n for n in trainable_names), \
        f"non-MTP params are trainable: {[n for n in trainable_names if 'mtp_heads' not in n]}"
    n_mtp = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[train_mtp] MTP params: {n_mtp:,} ({n_mtp / n_total * 100:.3f}% of {n_total:,})")
    print(f"[train_mtp] trainable tensors: {len(trainable_names)}")
    if args.check_only:
        print("[train_mtp] --check-only: OK")
        return

    if args.kl_only:
        print(f"[train_mtp] KL-ONLY training: depth loss = "
              f"lambda({args.distill_lambda}) * KL(top-{args.distill_topk}); "
              f"hard-CE anchor excluded from the gradient")
    if args.distill_topk > 0:
        print(f"[train_mtp] self-distillation: top-k={args.distill_topk} "
              f"lambda={args.distill_lambda} (teacher = frozen backbone's own "
              f"distribution at the aligned position)")
    if lk_mode:
        print(f"[train_mtp] LK loss ({args.mtp_loss}): MTP depths trained on "
              f"alpha = sum min(p, q) = 1 - TV(p, q) vs the frozen backbone "
              f"teacher (full-vocab; the direct acceptance-rate objective); "
              f"eta={args.lk_eta}")

    # ---- data ------------------------------------------------------------------
    # Three mutually exclusive modes: a separate --val-data file, an in-file
    # hold-out split, or no validation at all.
    rng = random.Random(args.seed + 1)
    if resume is not None and resume.get("rng_state") is not None:
        try:
            rng.setstate(resume["rng_state"])
            print("[train_mtp] resumed data-sampling RNG state (continues the same sample stream)")
        except Exception as e:
            print(f"[train_mtp] WARNING: sampling RNG state not restored: {e}")
    vds = None
    val_offsets: list[int] = []
    n_total = count_tokens(args.data, args.data_dtype)
    args._val_range = None      # recorded in the checkpoint meta by _save

    def _announce(msg: str) -> None:
        # Split information goes to the log as well as stdout.  The cut is
        # deterministic, but only if you also know the fraction it came from --
        # which is why the range itself is recorded rather than just the input.
        print(f"[train_mtp] {msg}")
        log_f.write(f"[train_mtp] {msg}\n")
        log_f.flush()

    if args.val_data:
        ds = BinDataset(args.data, args.ctx_len, args.data_dtype)
        vds = BinDataset(args.val_data, args.ctx_len, args.data_dtype)
        _announce(f"data: {args.data} ({ds.n_tokens:,} tokens, dtype {args.data_dtype})")
        _announce(f"val : {args.val_data} ({vds.n_tokens:,} tokens, separate file)")
    elif args.val_fraction > 0.0:
        if not 0.0 < args.val_fraction < 1.0:
            raise ValueError(f"--val-fraction must be in (0, 1), got {args.val_fraction}")
        cut = val_cut(args.data, args.data_dtype, n_total, args.val_fraction)
        ds = BinDataset(args.data, args.ctx_len, args.data_dtype, 0, cut)
        vds = BinDataset(args.data, args.ctx_len, args.data_dtype, cut, n_total)
        args._val_range = {
            "data": args.data,
            "val_fraction": args.val_fraction,
            "start": int(cut),
            "stop": int(n_total),
            "tokens": int(vds.n_tokens),
        }
        _announce(f"data: {args.data}[0:{cut:,}] "
                  f"({ds.n_tokens:,} tokens, dtype {args.data_dtype})")
        _announce(f"val : {args.data}[{cut:,}:{n_total:,}] "
                  f"({vds.n_tokens:,} tokens, {100 * vds.n_tokens / n_total:.1f}% held out)")
        _announce(f"val range is deterministic; recover it with "
                  f"`python scripts/extract_val.py --data {args.data} "
                  f"--data-dtype {args.data_dtype} --val-fraction {args.val_fraction}`")
    else:
        ds = BinDataset(args.data, args.ctx_len, args.data_dtype)
        _announce(f"data: {args.data} ({ds.n_tokens:,} tokens, dtype {args.data_dtype})")

    if vds is not None:
        val_offsets = _all_offsets(vds.n_tokens, args.ctx_len, args.val_chunks)
        _announce(f"val: {len(val_offsets)} chunks of {args.ctx_len}")

    # ---- trackio (optional) -----------------------------------------------------
    # Buffered, checkpoint-gated logging: records only reach trackio when a
    # checkpoint is actually saved (see _save), so the logged step range always
    # ends at a real checkpoint.  Restarting from that checkpoint resumes the
    # SAME trackio run (resume="allow" inside TrackioRun, keyed on the run
    # name) and keeps logging with explicit step numbers -- no orphaned rows
    # beyond the checkpoint ever exist, so nothing has to be deleted.
    tracker = TrackioRun(
        args.trackio_project,
        name=args.trackio_run or run_name,
        config=dict(vars(args)),
        tag="train_mtp",
    )
    tracker.buffer(run_start_rec)

    # ---- optimizer -------------------------------------------------------------
    opt = torch.optim.AdamW(
        params, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.weight_decay
    )
    if resume is not None and resume.get("optimizer") is not None:
        try:
            n_saved = sum(len(g["params"]) for g in resume["optimizer"]["param_groups"])
            if n_saved != len(params):
                raise ValueError(
                    f"saved optimizer tracks {n_saved} params, current model has "
                    f"{len(params)} (--num-mtp / --shared-head mismatch?)")
            opt.load_state_dict(resume["optimizer"])
            print(f"[train_mtp] resumed AdamW moments for all {len(params)} tensors")
        except Exception as e:
            print(f"[train_mtp] WARNING: optimizer state NOT restored ({e}); "
                  f"Adam moments restart from zero (weights and LR schedule unaffected)")
    
    model.train()
    t0 = time.time()
    tokens_seen = start_tokens
    B = max(1, args.batch_size)
    print(f"[train_mtp] batching: {B} seqs/micro-step x {args.grad_acc} micro-steps "
          f"x {args.ctx_len} ctx = {B * args.grad_acc * args.ctx_len:,} tokens/step "
          f"(VRAM scales ~linearly with batch size)")
    for step in range(start_step, args.steps):
        lr = cosine_lr(step, args.steps, args.lr, args.lr_final, args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        keys = ["total", "main"] + [f"mtp_{d}" for d in range(1, args.num_mtp + 1)]
        if lk_mode == "hybrid" or args.distill_topk > 0:
            keys += [f"kl_{d}" for d in range(1, args.num_mtp + 1)]
        if lk_mode:
            keys += [f"alpha_{d}" for d in range(1, args.num_mtp + 1)]
        running = {k: 0.0 for k in keys}
        step_tokens = 0

        for _ in range(args.grad_acc):
            xs, ys = [], []
            for _ in range(B):
                x, y = ds.sample(rng)
                xs.append(x)
                ys.append(y)
            x = torch.stack(xs).to(args.device, non_blocking=True)
            y = torch.stack(ys).to(args.device, non_blocking=True)
            try:
                loss, parts = model.forward_loss(
                    x, y, decay=args.decay, ignore_index=args.ignore_index,
                    return_parts=True,
                    include_main=(not args.skip_main_ce) and not args.kl_only,
                    distill_topk=args.distill_topk, distill_lambda=args.distill_lambda,
                    kl_only=args.kl_only, lk_mode=lk_mode, lk_eta=args.lk_eta,
                    lk_teacher_cache=args.lk_teacher_cache,
                )
                (loss / args.grad_acc).backward()
            except torch.cuda.OutOfMemoryError:
                raise SystemExit(
                    f"[train_mtp] CUDA OOM at batch_size={B}, ctx_len={args.ctx_len}. "
                    f"VRAM scales ~linearly with batch; reduce --batch-size "
                    f"(or --ctx-len) and raise --grad-acc to compensate."
                )
            for k in running:
                if k in parts:
                    running[k] += parts[k].item() / args.grad_acc
            step_tokens += x.numel()

        gnorm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        # detect non-finite grads and skip the step
        if torch.isfinite(gnorm):
            opt.step()
        else:
            print(f"[train_mtp] step {step}: non-finite grad norm {gnorm}; skipping step")
            opt.zero_grad(set_to_none=True)

        tokens_seen += step_tokens

        if step % args.log_every == 0 or step == args.steps - 1:
            dt = max(1e-6, time.time() - t0)
            dep = " ".join(f"d{d + 1}={running[f'mtp_{d + 1}']:.4f}" for d in range(args.num_mtp))
            alphaf = (" " + " ".join(f"a{d + 1}={running[f'alpha_{d + 1}']:.4f}"
                                     for d in range(args.num_mtp))) if lk_mode else ""
            klf = (" " + " ".join(f"kl{d + 1}={running[f'kl_{d + 1}']:.4f}"
                                  for d in range(args.num_mtp))
                   ) if f"kl_1" in running else ""
            main_field = ("" if args.skip_main_ce or args.kl_only
                          else f" main {running['main']:.4f}")
            msg = (f"step {step:6d}/{args.steps} lr {lr:.2e} "
                   f"loss {running['total']:.4f}{main_field} {dep}{klf}{alphaf} "
                   f"|g| {float(gnorm):.3f} {(tokens_seen - start_tokens) / dt / 1000:.1f}Ktok/s")
            print(f"[train_mtp] {msg}")
            log_f.write(msg + "\n")
            log_f.flush()
            rec = {"t": _dt.datetime.now().isoformat(timespec="seconds"),
                   "type": "train", "src": "live", "run": run_name,
                   "run_idx": run_idx, "step": step, "steps": args.steps,
                   "lr": lr, "total": running["total"],
                   "gnorm": float(gnorm), "tokens_seen": tokens_seen,
                   "ktok_s": (tokens_seen - start_tokens) / dt / 1000}
            if not args.skip_main_ce and not args.kl_only:
                rec["main"] = running["main"]
            for d in range(args.num_mtp):
                rec[f"mtp{d + 1}"] = running[f"mtp_{d + 1}"]
                if f"kl_{d + 1}" in running:
                    rec[f"kl{d + 1}"] = running[f"kl_{d + 1}"]
                if f"alpha_{d + 1}" in running:
                    rec[f"alpha{d + 1}"] = running[f"alpha_{d + 1}"]
            _emit_metrics(args.out, rec)
            tracker.buffer(rec)

        if vds is not None and args.val_every > 0 and (step + 1) % args.val_every == 0:
            vp = _validate(model, vds, val_offsets, args.batch_size,
                           args.ignore_index, args.device,
                           distill_topk=args.distill_topk,
                           distill_lambda=args.distill_lambda,
                           kl_only=args.kl_only, lk_mode=lk_mode,
                           lk_eta=args.lk_eta,
                           lk_teacher_cache=args.lk_teacher_cache)
            # decay=1.0 -> main + all depths uniformly; report each component so
            # the val numbers line up with the per-depth train numbers.
            vdep = " ".join(f"vd{d + 1}={vp[f'mtp_{d + 1}']:.4f}"
                            for d in range(args.num_mtp))
            vkl = (" " + " ".join(f"vkl{d + 1}={vp[f'kl_{d + 1}']:.4f}"
                                  for d in range(args.num_mtp))
                   ) if "kl_1" in vp else ""
            valpha = (" " + " ".join(f"va{d + 1}={vp[f'alpha_{d + 1}']:.4f}"
                                     for d in range(args.num_mtp))
                      ) if "alpha_1" in vp else ""
            msg = (f"step {step:6d} VAL loss {vp['total']:.4f} "
                   f"main {vp['main']:.4f} {vdep}{vkl}{valpha}")
            print(f"[train_mtp] {msg}")
            log_f.write(msg + "\n")
            log_f.flush()
            rec = {"t": _dt.datetime.now().isoformat(timespec="seconds"),
                   "type": "val", "src": "live", "run": run_name,
                   "run_idx": run_idx, "step": step, "steps": args.steps,
                   "total": vp["total"]}
            if "main" in vp:
                rec["main"] = vp["main"]
            for d in range(args.num_mtp):
                rec[f"mtp{d + 1}"] = vp[f"mtp_{d + 1}"]
                if f"kl_{d + 1}" in vp:
                    rec[f"kl{d + 1}"] = vp[f"kl_{d + 1}"]
                if f"alpha_{d + 1}" in vp:
                    rec[f"alpha{d + 1}"] = vp[f"alpha_{d + 1}"]
            _emit_metrics(args.out, rec)
            tracker.buffer(rec)

        if args.save_every > 0 and (step + 1) % args.save_every == 0:
            _save(model, args, step, tokens_seen, opt, rng, tracker)

    _save(model, args, args.steps, tokens_seen, opt, rng, tracker)
    log_f.write(f"DONE {_dt.datetime.now().isoformat()} tokens={tokens_seen}\n")
    log_f.close()
    tracker.finish()
    print(f"[train_mtp] done. saved {os.path.join(args.out, 'mtp_heads.pt')}")


def _save(model, args, step, tokens_seen, opt=None, rng=None, tracker=None):
    path = os.path.join(args.out, "mtp_heads.pt")
    meta = {
        "step": step,
        "tokens_seen": tokens_seen,
        "num_mtp": args.num_mtp,
        "ctx_len": args.ctx_len,
        "decay": args.decay,
        "ignore_index": args.ignore_index,
        "distill_topk": args.distill_topk,
        "distill_lambda": args.distill_lambda,
        "kl_only": args.kl_only,
        "mtp_loss": args.mtp_loss,
        "lk_eta": args.lk_eta,
        "lk_teacher_cache": args.lk_teacher_cache,
        "mtp_arch": args.mtp_arch,
        "eagle3_fusion": bool(getattr(args, "eagle3_fusion", False)),
        "lr": args.lr,
        "steps": args.steps,
        "warmup": args.warmup,
        "lr_final": args.lr_final,
        "pth": os.path.abspath(args.pth),
        "config": os.path.abspath(args.config),
        "data": os.path.abspath(args.data),
        "val_fraction": args.val_fraction,
        "val_range": getattr(args, "_val_range", None),
        "timestamp": _dt.datetime.now().isoformat(),
    }
    save_mtp_heads(model, path, meta)
    # Trainer state (optimizer moments + RNG streams) alongside the weights so
    # the next --load-mtp can pick up exactly where this run stopped.  Kept in
    # a separate file: mtp_heads.pt stays a pure-tensor payload that
    # eval/spec-decode readers can torch.load(..., weights_only=True).
    if opt is not None:
        trainer = {
            "step": step,
            "tokens_seen": tokens_seen,
            "optimizer": opt.state_dict(),
            "rng_state": rng.getstate() if rng is not None else None,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (torch.cuda.get_rng_state(args.device)
                               if args.device.startswith("cuda") else None),
            "saved_at": _dt.datetime.now().isoformat(),
        }
        torch.save(trainer, os.path.join(args.out, "trainer_state.pt"))
    print(f"[train_mtp] saved {path} (step {step})")
    # Only now -- after the checkpoint files are safely on disk -- do the
    # buffered records go to trackio.  If this run is killed later, the trackio
    # log still ends exactly at this checkpoint's step, so a restart from it
    # continues the same run with nothing to clean up.
    if tracker is not None:
        tracker.flush()


if __name__ == "__main__":
    main()
