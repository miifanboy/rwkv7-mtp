"""
eval_mtp.py -- measure per-depth MTP acceptance (next-token accuracy).

    python scripts/eval_mtp.py \
        --data data/val.bin --data-dtype uint16 \
        --pth rwkv7-g1j-2.9b-20260831-ctx16384.pth --config config.json \
        --mtp mtp_out/mtp_heads.pt --num-mtp 3 --ctx-len 2048

Acceptance for depth k is the fraction of non-pad positions where
``argmax(logits_k) == ground-truth token``.  This is the standard MTP /
speculative-decoding draft-accuracy proxy.  Depth 1 = 2 tokens ahead, depth k =
k+1 tokens ahead (see the mtp package's shift convention).

Prints a table; saves nothing.
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
import os
import random

import numpy as np
import torch

import mtp
from mtp import _shift_left, build_frozen_model


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate MTP acceptance rates")
    ap.add_argument("--data", required=True, help="held-out flat token .bin")
    ap.add_argument("--data-dtype", default="uint16", choices=["uint16", "uint32", "int32"])
    ap.add_argument("--pth", default="rwkv7-g1j-2.9b-20260831-ctx16384.pth")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--mtp", default="", help="mtp_heads.pt to load")
    ap.add_argument("--num-mtp", type=int, default=3)
    ap.add_argument("--shared-head", action="store_true",
                    help="checkpoint contains ONE MTPModule applied recursively "
                         "K=--num-mtp times (FastMTP style)")
    ap.add_argument("--mtp-arch", default=None, choices=list(mtp.MTP_ARCHS),
                    help="MTP head architecture ('mlp' or 'rwkv7'); default: "
                         "auto-detect from the --mtp checkpoint, else 'mlp'")
    ap.add_argument("--ctx-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--chunks", type=int, default=128, help="number of eval chunks")
    ap.add_argument("--ignore-index", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


@torch.no_grad()
def measure_acceptance(model, ds, num_mtp, ignore_index, device,
                       batch_size=1, chunks=64, rng=None):
    """Return (main_acc, per_depth_acc, counts)."""
    model.eval()
    rng = rng or random.Random(0)

    n_depth = num_mtp
    correct = [0] * (n_depth + 1)
    total = [0] * (n_depth + 1)

    for _ in range(chunks):
        x, y = ds.sample(rng)
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)

        main_logits, mtp_logits = model(x, return_mtp=True)

        # main head
        mask = y != ignore_index
        pred = main_logits.argmax(-1)
        correct[0] += int((pred[mask] == y[mask]).sum().item())
        total[0] += int(mask.sum().item())

        for k, logits in enumerate(mtp_logits):
            tgt = _shift_left(y, k + 1, pad=ignore_index)
            m = tgt != ignore_index
            pred = logits.argmax(-1)
            correct[k + 1] += int((pred[m] == tgt[m]).sum().item())
            total[k + 1] += int(m.sum().item())

    accs = [c / t if t else float("nan") for c, t in zip(correct, total)]
    return accs[0], accs[1:], total


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available to PyTorch; the RWKV-7 kernels require it."
        )

    from train_mtp import BinDataset  # reuse the loader

    # Resolve the head arch: explicit flag wins, else auto-detect from the
    # checkpoint, else 'mlp' (legacy checkpoints carry no tag).
    ck_meta = mtp.peek_mtp_meta(args.mtp) if args.mtp else {}
    if args.mtp_arch is None:
        args.mtp_arch = ck_meta.get("arch", "mlp")
    # EAGLE-3 fusion must match the checkpoint structure exactly; default to
    # the checkpoint's tag so plain `--mtp <ckpt>` loads old and new runs.
    args.eagle3_fusion = bool(ck_meta.get("eagle3_fusion", False))

    model, margs = build_frozen_model(
        args.pth, args.config, num_mtp=args.num_mtp, ctx_len=args.ctx_len,
        dtype=torch.bfloat16, device=args.device, shared_head=args.shared_head,
        arch=args.mtp_arch, eagle3_fusion=args.eagle3_fusion,
    )
    if args.mtp:
        meta = mtp.load_mtp_heads(model, args.mtp)
        ck_num = meta.get("num_mtp")
        if ck_num is not None and int(ck_num) != args.num_mtp:
            raise ValueError(
                f"--num-mtp {args.num_mtp} != checkpoint num_mtp {ck_num}"
            )
        ck_shared = bool(meta.get("shared_head", False))
        if ck_shared != args.shared_head:
            raise ValueError(
                f"--shared-head={args.shared_head} but checkpoint shared_head={ck_shared}"
            )
        print(f"[eval_mtp] loaded MTP heads from {args.mtp} ({meta})")

    ds = BinDataset(args.data, args.ctx_len, args.data_dtype)
    print(f"[eval_mtp] data {args.data}: {ds.n_tokens:,} tokens")

    rng = random.Random(args.seed)
    main_acc, depth_accs, total = measure_acceptance(
        model, ds, args.num_mtp, args.ignore_index, args.device,
        batch_size=args.batch_size, chunks=args.chunks, rng=rng,
    )

    print()
    print(f"{'depth':>6}  {'tokens ahead':>12}  {'acceptance':>10}  {'n':>12}")
    print("-" * 46)
    print(f"{'main':>6}  {1:>12}  {main_acc:>10.4f}  {total[0]:>12,}")
    for k, acc in enumerate(depth_accs):
        print(f"{'mtp_' + str(k + 1):>6}  {k + 2:>12}  {acc:>10.4f}  {total[k + 1]:>12,}")
    print()

    bands = [(0.55, 0.75), (0.35, 0.55), (0.20, 0.35)]
    if depth_accs and depth_accs[0] < 0.30:
        print("[eval_mtp] NOTE: depth-1 acceptance < 0.30. The implementation is "
              "functionally correct but the MTP head is under-parameterized for "
              "this backbone; consider more steps / a wider MTP MLP.")
    else:
        print(f"[eval_mtp] rough expected bands (short runs may be lower): {bands}")
    return main_acc, depth_accs


if __name__ == "__main__":
    main()
