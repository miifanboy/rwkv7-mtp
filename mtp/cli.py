"""Command-line helpers for the MTP package (``python -m mtp ...``)."""

from __future__ import annotations

import argparse

import torch

from .checkpoint import load_pth_state_dict
from .tensors import _shift_left


def _main() -> None:
    ap = argparse.ArgumentParser(description="MTP helpers")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ins = sub.add_parser("inspect", help="inspect a .pth checkpoint")
    p_ins.add_argument("pth")

    sub.add_parser("shift", help="run the shift-convention self-test")

    a = ap.parse_args()
    if a.cmd == "inspect":
        sd = load_pth_state_dict(a.pth)
        ks = list(sd.keys())
        n = sum(v.numel() for v in sd.values())
        print(f"keys: {len(ks)}")
        print(f"first: {ks[0]}  last: {ks[-1]}")
        print(f"dtype: {sd[ks[0]].dtype}  params: {n / 1e9:.3f}B")
        print(f"has head.weight: {'head.weight' in sd}  emb.weight: {'emb.weight' in sd}")
    elif a.cmd == "shift":
        run_shift_selftest()


def run_shift_selftest() -> None:
    t = torch.arange(20).reshape(2, 10)  # [[0..9], [10..19]]

    s1 = _shift_left(t, 1, pad=-1)
    assert torch.equal(s1[0, :9], t[0, 1:]), "shift-1 broken"
    assert s1[0, -1].item() == -1, "pad broken"

    s3 = _shift_left(t, 3, pad=-1)
    assert torch.equal(s3[0, :7], t[0, 3:])
    assert (s3[0, -3:] == -1).all()

    # Consistency of the MTP convention: input token is one step before target.
    idx = torch.arange(0, 12).unsqueeze(0)          # x = [0..11]
    targets = _shift_left(idx, 1, pad=-1)           # targets[i] = x[i+1]
    for k in range(3):
        inp = _shift_left(idx, k + 1, pad=0)        # x[i+k+1]
        tgt = _shift_left(targets, k + 1, pad=-1)   # x[i+k+2]
        direct_tgt = torch.cat(
            [idx[:, k + 2:], torch.full((1, k + 2), -1, dtype=idx.dtype)], dim=1
        )
        assert torch.equal(tgt, direct_tgt), f"target shift broken at depth {k}"
        assert torch.equal(inp[:, :11 - k - 1], idx[:, k + 1:11]), "input shift broken"
    print("shift ok")
