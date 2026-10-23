"""Rebuild <out>/metrics.jsonl from the human-readable train_log.txt.

This is the backfill path for runs that started before metrics.jsonl existed
(the first ~25% of the mtp_out_cont run).  It is fully idempotent: it parses
the *entire* text log and rewrites metrics.jsonl from scratch, so you can
re-run it at any time (mid-run, after a resume, or after the run finishes)
and always get a complete, deduplicated file.

Merge rule
----------
Any record already present in the existing metrics.jsonl that was written
live by train_mtp.py (``"src": "live"``) overrides the parsed record with the
same (run_idx, type, step) key -- live records carry exact full-precision
values and per-record timestamps, while text-log values are rounded
(4 decimals for losses, 3 for grad norm) and only have timestamps at
NEW RUN / DONE markers.

Usage
-----
    python scripts/parse_train_log.py --log mtp_out_cont/train_log.txt
    python scripts/parse_train_log.py --log mtp_out_cont/train_log.txt --out plot/metrics.jsonl

Schema of emitted records (same as the live trainer):
    {"t": iso-timestamp-or-null, "type": "train"|"val"|"run_start",
     "src": "parse", "run": <out-dir name>, "run_idx": <1-based NEW RUN index>,
     "step": int, "steps": int, "lr": float, "total": float, "main": float,
     "mtp1..N": float, "kl1..N": float, "gnorm": float, "ktok_s": float,
     "tokens_seen": int (only when exactly recoverable, see below),
     "args": {...}  (run_start records only)}

Notes
-----
* ``tokens_seen`` is only emitted when the run's first logged step is 0 (fresh
  start), where it is exactly (step + 1) * batch_size * ctx_len * grad_acc.
  For resumed segments the text log does not carry the resume offset, so the
  field is omitted there (live JSONL records always have the true value).
* ``ktok_s`` is not emitted by the parser (no per-line timestamps to derive
  the elapsed time from); live records have it.
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
import ast
import json
import os
import re
import sys

_NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

# step   7770/32000 lr 8.91e-05 loss 1.1556 main 0.4062 d1=... kl1=... |g| 1.133 1.8Ktok/s
_TRAIN_RE = re.compile(
    rf"^\s*step\s+(\d+)/(\d+)\s+lr\s+({_NUM})\s+loss\s+({_NUM})(.*)$")
# step   7799 VAL loss 1.6090 main 0.4984 vd1=1.6644 vd2=2.0290 vd3=2.2441
_VAL_RE = re.compile(rf"^\s*step\s+(\d+)\s+VAL\s+loss\s+({_NUM})(.*)$")
# NEW RUN 2026-10-04T13:04:09.600833 {'data': ..., 'out': ...}
_NEW_RUN_RE = re.compile(r"^NEW RUN\s+(\S+)\s+(\{.*\})\s*$")
_KV_RE = re.compile(rf"(\w+)=({_NUM})")
_MAIN_RE = re.compile(rf"\bmain\s+({_NUM})\b")
_GNORM_RE = re.compile(rf"\|g\|\s+({_NUM})")
_KTOK_RE = re.compile(rf"({_NUM})Ktok/s")
# Word-boundary numbers not attached to a key= or Ktok/s (the trailing rate on
# old lines is handled by _KTOK_RE, so nothing else should be picked up).


def _keyval_pairs(rest: str) -> dict[str, float]:
    """Extract d1=/kl1=/vd1= style pairs and remap them to schema names."""
    out: dict[str, float] = {}
    for k, v in _KV_RE.findall(rest):
        if k.startswith("d") and k[1:].isdigit():
            out[f"mtp{k[1:]}"] = float(v)
        elif k.startswith("vd") and k[2:].isdigit():
            out[f"mtp{k[2:]}"] = float(v)
        elif k.startswith("vkl") and k[3:].isdigit():
            out[f"kl{k[3:]}"] = float(v)
        elif k.startswith("kl") and k[2:].isdigit():
            out[f"kl{k[2:]}"] = float(v)
        elif k.startswith("va") and k[2:].isdigit():
            out[f"alpha{k[2:]}"] = float(v)
        elif k.startswith("a") and k[1:].isdigit():
            out[f"alpha{k[1:]}"] = float(v)
        else:
            out[k] = float(v)
    return out


def parse_train_log(log_path: str) -> list[dict]:
    with open(log_path, encoding="utf-8") as f:
        lines = f.readlines()

    run_name = os.path.basename(os.path.normpath(os.path.dirname(log_path)))
    records: list[dict] = []
    run_idx = 0
    run_args: dict = {}
    run_start_t: str | None = None

    for line in lines:
        line = line.rstrip("\n")
        m = _NEW_RUN_RE.match(line)
        if m:
            run_idx += 1
            run_start_t = m.group(1)
            try:
                # The header is a Python dict repr (True/False, not true/false),
                # so ast.literal_eval is the correct parser -- json.loads fails
                # on the booleans.
                run_args = ast.literal_eval(m.group(2))
            except Exception:
                run_args = {}
            records.append({"t": run_start_t, "type": "run_start", "src": "parse",
                            "run": run_name, "run_idx": run_idx, "args": run_args})
            continue
        if not line.startswith("step"):
            continue  # [train_mtp] announce lines, DONE line, etc.

        m = _TRAIN_RE.match(line)
        if m:
            step, steps = int(m.group(1)), int(m.group(2))
            rest = m.group(5)
            rec = {"type": "train", "src": "parse", "run": run_name,
                   "run_idx": run_idx, "step": step, "steps": steps,
                   "lr": float(m.group(3)), "total": float(m.group(4))}
            if (mm := _MAIN_RE.search(rest)):
                rec["main"] = float(mm.group(1))
            rec.update(_keyval_pairs(rest))
            if (mm := _GNORM_RE.search(rest)):
                rec["gnorm"] = float(mm.group(1))
            if (mm := _KTOK_RE.search(rest)):
                rec["ktok_s"] = float(mm.group(1))
        else:
            m = _VAL_RE.match(line)
            if not m:
                continue
            step = int(m.group(1))
            rest = m.group(3)
            rec = {"type": "val", "src": "parse", "run": run_name,
                   "run_idx": run_idx, "step": step,
                   "total": float(m.group(2))}
            if (mm := _MAIN_RE.search(rest)):
                rec["main"] = float(mm.group(1))
            rec.update(_keyval_pairs(rest))

        # run_start timestamp is the only wall-clock anchor the text log has;
        # reuse it so records are never missing the t field entirely.
        rec["t"] = run_start_t
        records.append(rec)

    # tokens_seen: exact when the file's very first logged train step is 0 (a
    # fresh start), because checkpoint resumes continue the token counter:
    # tokens_seen at step s is then exactly (s + 1) * tokens-per-step for the
    # whole file.  Only safe if every NEW RUN section used the same batch/ctx/
    # grad-acc; otherwise omit the field rather than emit a wrong number.
    first_train = next((r for r in records if r.get("type") == "train"), None)
    if first_train is not None and first_train["step"] == 0:
        tps_by_run: dict[int, int] = {}
        for rec in records:
            if rec["type"] == "run_start":
                a = rec["args"]
                tps_by_run[rec["run_idx"]] = (int(a.get("batch_size", 1))
                                              * int(a.get("ctx_len", 1))
                                              * int(a.get("grad_acc", 1)))
        tps_values = {tps_by_run.get(r["run_idx"]) for r in records
                      if r.get("type") == "train"}
        if len(tps_values) == 1 and None not in tps_values:
            tps = tps_values.pop()
            for rec in records:
                if rec.get("type") == "train":
                    rec["tokens_seen"] = (rec["step"] + 1) * tps

    return records


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True, help="path to train_log.txt")
    ap.add_argument("--out", default=None,
                    help="output metrics.jsonl path (default: <log dir>/metrics.jsonl)")
    args = ap.parse_args(argv)

    out_path = args.out or os.path.join(os.path.dirname(args.log), "metrics.jsonl")
    parsed = parse_train_log(args.log)

    # Merge: live records already in an existing metrics.jsonl win over parsed
    # ones (full precision + real timestamps), keyed by (run_idx, type, step).
    live: dict[tuple, dict] = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn tail line from a crash; the text log has it
                if rec.get("src") == "live":
                    key = (rec.get("run_idx"), rec.get("type"), rec.get("step"))
                    live[key] = rec

    merged: dict[tuple, dict] = {}
    for rec in parsed:
        key = (rec.get("run_idx"), rec.get("type"), rec.get("step"))
        merged[key] = {**rec, **live.get(key, {})}
    # keep live-only records (e.g. metrics fields the text log cannot express)
    for key, rec in live.items():
        merged.setdefault(key, rec)

    out = sorted(merged.values(),
                 key=lambda r: (r.get("run_idx") or 0,
                                r.get("step") if r.get("step") is not None else -1,
                                0 if r.get("type") == "run_start" else 1,
                                r.get("type") or ""))
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_train = sum(1 for r in out if r.get("type") == "train")
    n_val = sum(1 for r in out if r.get("type") == "val")
    n_live = sum(1 for r in out if r.get("src") == "live")
    print(f"[parse_train_log] wrote {len(out)} records -> {out_path} "
          f"(train={n_train} val={n_val}, {n_live} from live JSONL, "
          f"rest parsed from {args.log})")


if __name__ == "__main__":
    sys.exit(main())