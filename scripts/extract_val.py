"""
extract_val.py -- materialise the validation split used by train_mtp.py.

``train_mtp.py --val-fraction`` holds out the *tail* of the training ``.bin``
in-file, so no ``val.bin`` is ever written.  ``eval_mtp.py`` and
``spec_decode.py`` both want a path, so this carves the identical range out.

The cut is deterministic (see ``train_mtp.val_cut``), so re-running with the
same ``--data``/``--val-fraction`` reproduces the split from your training run
exactly.  ``--start``/``--stop`` override it if you would rather be explicit.

    python scripts/extract_val.py --data data/train.bin --val-fraction 0.05 \
        --out data/val.bin
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
import json
import os

import numpy as np

from train_mtp import _DTYPES, count_tokens, val_cut


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Write the train_mtp val split to a .bin")
    ap.add_argument("--data", required=True, help="the training .bin that was split")
    ap.add_argument("--data-dtype", default="uint16", choices=list(_DTYPES))
    ap.add_argument("--val-fraction", type=float, default=0.0,
                    help="must match the training run (the tail fraction held out)")
    ap.add_argument("--start", type=int, default=-1, help="explicit cut (overrides --val-fraction)")
    ap.add_argument("--stop", type=int, default=-1, help="explicit end (default: end of file)")
    ap.add_argument("--out", default="data/val.bin")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"not found: {args.data}")

    n_total = count_tokens(args.data, args.data_dtype)
    stop = n_total if args.stop < 0 else args.stop
    if args.start >= 0:
        cut = args.start
    elif args.val_fraction > 0.0:
        cut = val_cut(args.data, args.data_dtype, n_total, args.val_fraction)
    else:
        raise SystemExit(
            "need either --val-fraction > 0 or an explicit --start"
        )
    if not (0 <= cut < stop <= n_total):
        raise SystemExit(f"bad range [{cut}, {stop}) for {n_total} tokens")

    dtype = _DTYPES[args.data_dtype]
    src = np.memmap(args.data, dtype=dtype, mode="r")
    chunk = np.asarray(src[cut:stop])

    # Sanity: the cut must be a document boundary (token 0 = EOS).
    at_eos = cut > 0 and int(src[cut - 1]) == 0

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    chunk.tofile(args.out)

    meta = {
        "source": os.path.abspath(args.data),
        "source_tokens": int(n_total),
        "start": int(cut),
        "stop": int(stop),
        "tokens": int(chunk.size),
        "fraction": round(chunk.size / n_total, 6),
        "dtype": args.data_dtype,
        "cut_at_eos": bool(at_eos),
        "trailing_tokens_before_cut": None,
    }
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[extract_val] {args.data}[{cut:,}:{stop:,}] -> {args.out}")
    print(f"[extract_val] {chunk.size:,} tokens ({100 * chunk.size / n_total:.2f}% "
          f"of {n_total:,}), dtype {args.data_dtype}, cut_at_eos={at_eos}")


if __name__ == "__main__":
    main()
