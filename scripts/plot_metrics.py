"""Render metrics.jsonl (written live by train_mtp.py or rebuilt from
train_log.txt by parse_train_log.py) into a multi-panel PNG.

Usage
-----
    python scripts/plot_metrics.py --metrics mtp_out_cont/metrics.jsonl
    python scripts/plot_metrics.py --metrics mtp_out_cont/metrics.jsonl --x tokens --smooth 0.9
    python scripts/plot_metrics.py --metrics mtp_out_cont/metrics.jsonl --split-runs --show

Panels: total loss, main CE, per-depth MTP loss, per-depth KL, grad norm, LR,
throughput.  Train curves are drawn as faint raw + bold EMA-smoothed lines;
validation points are overlaid as scatter markers on the matching series.

Restart handling: a train_log.txt can contain several NEW RUN sections
(crash/resume).  By default the sections are merged into one continuous curve,
deduplicating re-executed steps (the resume re-runs steps after the last
checkpoint; the later record wins).  --split-runs draws each section as its own
series instead, and restart boundaries are marked with vertical dashed lines.
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
import sys
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def load_records(path: str) -> list[dict]:
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # torn tail line from a crash
    return recs


def dedupe(recs: list[dict]) -> list[dict]:
    """Keep the latest record per (type, step) across all NEW RUN sections.

    Input must already be ordered by (run_idx, step) -- parse_train_log.py and
    the trainer both append chronologically, so later records are re-executions
    or continuations of earlier ones.
    """
    out: dict[tuple, dict] = {}
    for r in recs:
        out[(r.get("type"), r.get("step"))] = r
    return sorted(out.values(), key=lambda r: (r.get("step") or 0,
                                               0 if r.get("type") == "train" else 1))


def ema(y: np.ndarray, alpha: float) -> np.ndarray:
    if len(y) == 0:
        return y
    s = np.empty_like(y, dtype=float)
    s[0] = y[0]
    for i in range(1, len(y)):
        s[i] = alpha * s[i - 1] + (1.0 - alpha) * y[i]
    return s


PANELS = [  # (title, [(key, label), ...], log_y)
    ("total loss", [("total", "train"), ("total", "val")], False),
    ("main CE", [("main", "train"), ("main", "val")], False),
    ("MTP loss / depth", [("mtp1", "train"), ("mtp2", "train"), ("mtp3", "train"),
                          ("mtp1", "val"), ("mtp2", "val"), ("mtp3", "val")], False),
    ("KL distill / depth", [("kl1", "train"), ("kl2", "train"), ("kl3", "train")], False),
    ("grad norm", [("gnorm", "train")], False),
    ("lr", [("lr", "train")], True),
    ("Ktok/s", [("ktok_s", "train")], False),
]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", required=True, help="path to metrics.jsonl")
    ap.add_argument("--out", default=None,
                    help="output PNG (default: <metrics dir>/metrics.png)")
    ap.add_argument("--show", action="store_true", help="open an interactive window")
    ap.add_argument("--smooth", type=float, default=0.9,
                    help="EMA factor for train curves, 0..1 (0 = raw only; default 0.9)")
    ap.add_argument("--x", choices=["step", "tokens"], default="step",
                    help="x-axis: optimizer step or tokens_seen (falls back to step "
                         "for records without tokens_seen)")
    ap.add_argument("--split-runs", action="store_true",
                    help="draw each NEW RUN section as a separate series instead of "
                         "merging crash-resumes into one continuous curve")
    args = ap.parse_args(argv)

    recs = load_records(args.metrics)
    train_recs = [r for r in recs if r.get("type") == "train"]
    val_recs = [r for r in recs if r.get("type") == "val"]
    runs = sorted({r.get("run_idx", 1) for r in train_recs})
    run_name = next((r.get("run") for r in recs if r.get("run")), os.path.basename(args.metrics))

    if not train_recs and not val_recs:
        raise SystemExit(f"[plot_metrics] no train/val records in {args.metrics}")

    # restart boundaries: first step of every section after the first
    first_steps = []
    for i in runs[1:]:
        steps = [r["step"] for r in train_recs
                 if r.get("run_idx", 1) == i and r.get("step") is not None]
        if steps:
            first_steps.append(min(steps))
    restarts = sorted(set(first_steps)) if len(runs) > 1 else []

    if args.x == "tokens":
        def xval(r):
            return r.get("tokens_seen")
    else:
        def xval(r):
            return r.get("step")

    restart_x = []
    if restarts:
        by_step = {r.get("step"): r for r in train_recs}
        restart_x = [xval(by_step[s]) for s in restarts
                     if s in by_step and xval(by_step[s]) is not None]

    n_panels = sum(1 for _, keys, _ in PANELS
                   if any(k in r for r in train_recs + val_recs for k, _ in keys))
    ncols = 2 if n_panels <= 4 else 3
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 3.2 * nrows),
                             squeeze=False, sharex=True)
    fig.suptitle(f"{run_name}  (through step "
                 f"{max((r.get('step') or 0) for r in train_recs + val_recs)})",
                 fontsize=11)

    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    p = 0
    for title, keys, logy in PANELS:
        if not any(k in r for r in train_recs + val_recs for k, _ in keys):
            continue
        ax = axes[p // ncols][p % ncols]
        p += 1
        ax.set_title(title, fontsize=10)
        if logy:
            ax.set_yscale("log")

        if args.split_runs:
            series_groups = [([r for r in train_recs if r.get("run_idx", 1) == i],
                              [r for r in val_recs if r.get("run_idx", 1) == i])
                             for i in runs]
        else:
            series_groups = [(dedupe(train_recs), dedupe(val_recs))]

        ci = 0
        for gi, (tr, va) in enumerate(series_groups):
            suffix = f" (run {runs[gi]})" if args.split_runs and len(runs) > 1 else ""
            for key, kind in keys:
                src = tr if kind == "train" else va
                pts = [(xval(r), r[key]) for r in src
                       if key in r and r[key] is not None and xval(r) is not None]
                if not pts:
                    continue
                xs = [x for x, _ in pts]
                ys = [y for _, y in pts]
                label = f"{key}{suffix}" if len(keys) > 1 or suffix else key
                if kind == "train":
                    c = palette[ci % len(palette)]
                    ci += 1
                    ax.plot(xs, ys, color=c, alpha=0.25, lw=0.8)
                    if args.smooth > 0:
                        ax.plot(xs, ema(np.array(ys), args.smooth), color=c, lw=1.6,
                                label=label)
                    else:
                        ax.plot(xs, ys, color=c, lw=1.2, label=label)
                else:
                    # val markers inherit the color of the matching train series
                    c = palette[(ci - 1) % len(palette)] if ci else palette[0]
                    ax.plot(xs, ys, "o", ms=3.5, color=c, alpha=0.9,
                            label=f"{label} val", mfc="none")
        for xv in restart_x:
            ax.axvline(xv, color="gray", ls="--", lw=0.7, alpha=0.6)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    # hide unused axes
    for r_ in range(nrows):
        for c_ in range(ncols):
            if (r_ * ncols + c_) >= n_panels:
                axes[r_][c_].set_visible(False)

    for c_ in range(ncols):
        axes[nrows - 1][c_].set_xlabel("tokens" if args.x == "tokens" else "step",
                                       fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path = args.out or os.path.join(os.path.dirname(args.metrics), "metrics.png")
    fig.savefig(out_path, dpi=140)
    print(f"[plot_metrics] wrote {out_path}")
    if args.show:
        matplotlib.use("TkAgg" if os.name == "nt" else matplotlib.get_backend())
        plt.show()


if __name__ == "__main__":
    sys.exit(main())