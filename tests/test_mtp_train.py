"""
one-batch overfit test on the real (frozen) backbone.

    python tests/test_mtp_train.py --ctx-len 512 --steps 60 --lr 2e-3

Proves:
  * the frozen backbone produces hidden states through the CUDA kernels,
  * gradients flow into every MTP head,
  * no gradient reaches any backbone parameter,
  * the combined loss drops >50% by memorizing a single batch.
"""

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

import torch

from mtp import MTP_ARCHS, build_frozen_model, mtp_parameters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", default="rwkv7-g1j-2.9b-20260831-ctx16384.pth")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--num-mtp", type=int, default=3)
    ap.add_argument("--mtp-arch", default=None, choices=list(MTP_ARCHS),
                    help="MTP head architecture: 'mlp' (default) or 'rwkv7' "
                         "(one full RWKV-7 block; needs CUDA)")
    ap.add_argument("--ctx-len", type=int, default=512)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=None,
                    help="default: 2e-3 for the mlp head, 3e-4 for rwkv7 (the "
                         "block's fresh random-init projections are much "
                         "smaller than the mlp head's and diverge at 2e-3; "
                         "production training uses 1e-4 + warmup anyway)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    arch = a.mtp_arch or "mlp"
    if a.lr is None:
        a.lr = 2e-3 if arch == "mlp" else 3e-4
    model, args = build_frozen_model(
        a.pth, a.config, num_mtp=a.num_mtp, ctx_len=a.ctx_len,
        dtype=torch.bfloat16, device=a.device, arch=arch,
    )
    print(f"[test_mtp_train] arch={arch} lr={a.lr:g}")

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("mtp_heads" in n for n in trainable)

    V = args.vocab_size
    x = torch.randint(1, V, (1, a.ctx_len), device=a.device)
    y = torch.randint(1, V, (1, a.ctx_len), device=a.device)

    params = mtp_parameters(model)
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), eps=1e-8)
    model.train()

    first = None
    parts = None
    for s in range(a.steps):
        opt.zero_grad(set_to_none=True)
        loss, parts = model.forward_loss(
            x, y, decay=0.8, ignore_index=0, return_parts=True
        )
        loss.backward()

        if s == 0:
            for n, p in model.named_parameters():
                if "mtp_heads" in n:
                    assert p.grad is not None, f"MTP param got no grad: {n}"
                else:
                    assert p.grad is None, f"backbone param got grad: {n}"
            first = {k: v.item() for k, v in parts.items()}

        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if s % 10 == 0 or s == a.steps - 1:
            dep = " ".join(f"{k}={v.item():.3f}" for k, v in parts.items() if k != "total")
            print(f"step {s:3d}  total={parts['total'].item():.3f}  {dep}")

    final = parts["total"].item()
    drop = 100.0 * (1.0 - final / first["total"])
    print(f"\ncombined loss {first['total']:.3f} -> {final:.3f}  ({drop:.1f}% drop)")
    assert final < 0.5 * first["total"], "loss did not drop by 50%"
    print("ONE-BATCH OVERFIT OK")


if __name__ == "__main__":
    main()
