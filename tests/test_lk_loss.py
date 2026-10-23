"""test_lk_loss.py -- CPU correctness check for _FusedLK / _lk_loss (mtp.py).

Compares the fused chunked implementation against the naive full-tensor
reference from docs/lk-loss.md:

    alpha = sum min(p, q) = 1 - TV(p, q)
    hybrid   :  L = lam * KL(p||q) + (1-lam) * TV(p,q),  lam = exp(-eta*mean(alpha))
    lk_alpha :  L = -log(alpha + eps)

Checks loss values, gradients w.r.t. the draft hiddens (vs autograd through a
naive logits path), ignore_index masking, and chunk-size invariance.
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

import torch
import torch.nn.functional as F

import mtp
from mtp import _lk_loss, LK_EPS

torch.manual_seed(0)

B, T, C, V = 2, 8, 16, 97
IGNORE = 0


def reference(z, t_h, weight, targets, mode, eta, eps=LK_EPS):
    """Full-tensor, autograd-friendly reference (docs/lk-loss.md pseudocode)."""
    z_logits = F.linear(z, weight).float()
    p_logits = F.linear(t_h, weight).float()
    q = F.softmax(z_logits, dim=-1)
    p = F.softmax(p_logits, dim=-1)
    valid = targets != IGNORE
    q, p = q[valid], p[valid]
    alpha = torch.minimum(p, q).sum(dim=-1)
    tv = 1.0 - alpha
    kl = (p * (p.clamp_min(1e-30).log() - q.clamp_min(1e-30).log())).sum(dim=-1)
    if mode == "hybrid":
        # sg[alpha]: the schedule is a stop-gradient (docs/lk-loss.md §3.1)
        lam = torch.exp(-eta * alpha.mean().detach())
        return (lam * kl + (1.0 - lam) * tv).mean(), alpha.mean(), kl.mean(), tv.mean()
    return (-torch.log(alpha + eps)).mean(), alpha.mean(), kl.mean(), tv.mean()


def run_case(mode, eta, chunk, with_ignored=True):
    z = torch.randn(B, T, C, dtype=torch.float32, requires_grad=True)
    t_h = torch.randn(B, T, C, dtype=torch.float32)
    weight = torch.randn(V, C, dtype=torch.float32)
    targets = torch.randint(2, V, (B, T))
    if with_ignored:
        targets[0, 2] = IGNORE
        targets[1, 5] = IGNORE

    # fused
    loss_f, a_f, kl_f, tv_f = _lk_loss(z, t_h, weight, targets, IGNORE,
                                       mode, eta, chunk=chunk)
    g_f = torch.autograd.grad(loss_f, z)[0]

    # reference
    loss_r, a_r, kl_r, tv_r = reference(z, t_h, weight, targets, mode, eta)
    g_r = torch.autograd.grad(loss_r, z)[0]

    tag = f"[{mode} eta={eta} chunk={chunk} ignored={with_ignored}]"
    assert torch.allclose(loss_f, loss_r, rtol=1e-5, atol=1e-6), \
        f"{tag} loss {loss_f.item()} != {loss_r.item()}"
    checks = [("alpha", a_f, a_r), ("tv", tv_f, tv_r)]
    if mode == "hybrid":
        checks.append(("kl", kl_f, kl_r))
    for name, x, y in checks:
        assert torch.allclose(x, y, rtol=1e-5, atol=1e-6), \
            f"{tag} {name} {x.item()} != {y.item()}"
    # fp32 noise: the 1/(alpha + eps) division in lk_alpha amplifies softmax
    # rounding (~1e-5 abs); bf16 training makes this irrelevant.
    assert torch.allclose(g_f, g_r, rtol=1e-3, atol=1e-4), \
        f"{tag} grad mismatch (max abs diff {(g_f - g_r).abs().max().item()})"

    # ignore_index rows must receive exactly zero gradient
    ignored_mask = (targets == IGNORE)
    if ignored_mask.any():
        assert g_f[ignored_mask].abs().max().item() == 0.0, f"{tag} ignored rows got grad"
    print(f"  ok {tag}: loss={loss_f.item():.6f} alpha={a_f.item():.6f} "
          f"|dg|={g_f.abs().max().item():.6f}")


def run_alpha_identity():
    """alpha = 1 - TV must hold row-wise, and alpha in (0, 1]."""
    z = torch.randn(3, 5, C)
    t_h = torch.randn(3, 5, C)
    weight = torch.randn(V, C)
    targets = torch.randint(2, V, (3, 5))
    z_logits = F.linear(z, weight).float()
    q = F.softmax(z_logits, dim=-1)
    p = F.softmax(F.linear(t_h, weight).float(), dim=-1)
    alpha = torch.minimum(p, q).sum(-1)
    tv = 0.5 * (p - q).abs().sum(-1)
    assert torch.allclose(alpha, 1.0 - tv, atol=1e-6)
    assert (alpha > 0).all() and (alpha <= 1.0).all()
    print("  ok alpha == 1 - TV row-wise")


print("test_lk_loss:")
run_alpha_identity()
for mode, etas in (("hybrid", (3.0, 0.0, 10.0)), ("lk_alpha", (3.0,))):
    for eta in etas:
        run_case(mode, eta, chunk=7)
        run_case(mode, eta, chunk=1000)   # single chunk must agree
run_case("hybrid", 3.0, chunk=5, with_ignored=False)

# teacher-logits cache path must be equivalent to the hiddens path (B > 1 to
# exercise the batch-interleaved row indexing: flat row r = b*n + i reads
# table row b*T + i + offset).
def run_cache_case(mode):
    torch.manual_seed(1)
    Tb, n, off = 11, 9, 2          # teacher position of row i is i + off
    B, C = 3, 16
    z = torch.randn(B, n, C, dtype=torch.float32, requires_grad=True)
    t_h_full = torch.randn(B, Tb, C, dtype=torch.float32)
    weight = torch.randn(V, C, dtype=torch.float32)
    targets = torch.randint(2, V, (B, n))
    targets[1, 3] = IGNORE
    t_h = t_h_full[:, off:off + n]                       # aligned hiddens
    table = F.linear(t_h_full.reshape(-1, C), weight)    # (B*Tb, V) bf16-like

    loss_h, a_h, kl_h, tv_h = _lk_loss(z, t_h, weight, targets, IGNORE,
                                       mode, 3.0, chunk=7)
    g_h = torch.autograd.grad(loss_h, z)[0]
    loss_t, a_t, kl_t, tv_t = _lk_loss(z, None, weight, targets, IGNORE,
                                       mode, 3.0, t_logits=table,
                                       t_offset=off, t_T=Tb, chunk=7)
    g_t = torch.autograd.grad(loss_t, z)[0]
    assert torch.allclose(loss_h, loss_t, rtol=1e-6, atol=1e-6), \
        f"[cache {mode}] loss {loss_h.item()} != {loss_t.item()}"
    assert torch.allclose(a_h, a_t, rtol=1e-5, atol=1e-6)
    assert torch.allclose(g_h, g_t, rtol=1e-3, atol=1e-5), \
        f"[cache {mode}] grad diff {(g_h - g_t).abs().max().item()}"
    print(f"  ok teacher-cache == hiddens [{mode}]: loss={loss_t.item():.6f}")

run_cache_case("hybrid")
run_cache_case("lk_alpha")


def run_cuda_case(mode, use_table=True):
    """CUDA fused-kernel path must match the eager path (RWKV_LK_CUDA=0).

    Exercises BOTH teacher sources: the shared logits table (B=1 training,
    cache on) and the per-depth hiddens slice (B=4, cache over budget) -- the
    latter is what caught a leftover .float() returning fp32 to the bf16
    kernels.
    """
    if not torch.cuda.is_available():
        return
    ext = mtp._lk_cuda_ext()
    if ext is None:
        print("  SKIP cuda-vs-eager (kernel unavailable)")
        return
    torch.manual_seed(2)
    B, n, C, off, Tb = 3, 9, 16, 2, 11
    z = torch.randn(B, n, C, dtype=torch.bfloat16, requires_grad=True)
    t_h_full = torch.randn(B, Tb, C, dtype=torch.bfloat16)
    weight = torch.randn(V, C, dtype=torch.bfloat16)
    targets = torch.randint(2, V, (B, n))
    targets[1, 3] = IGNORE
    t_h = t_h_full[:, off:off + n]
    table = F.linear(t_h_full.reshape(-1, C), weight)

    def run(dev, force_eager):
        old = os.environ.get("RWKV_LK_CUDA")
        os.environ["RWKV_LK_CUDA"] = "0" if force_eager else "1"
        try:
            lk_mod = mtp.losses.lk
            lk_mod._LK_EXT_TRIED = lk_mod._LK_EXT_TRIED and not force_eager
            lk_mod._LK_EXT = None if force_eager else ext
            z_d = z.detach().to(dev).requires_grad_(True)
            loss, a, kl_m, tv_m = _lk_loss(
                z_d, None if use_table else t_h.to(dev), weight.to(dev),
                targets.to(dev), IGNORE, mode, 3.0,
                t_logits=table.to(dev) if use_table else None,
                t_offset=off if use_table else 0,
                t_T=Tb if use_table else 0, chunk=7)
            g = torch.autograd.grad(loss, z_d)[0]
            return loss.item(), a.item(), g.cpu()
        finally:
            os.environ["RWKV_LK_CUDA"] = old if old is not None else "1"
            lk_mod = mtp.losses.lk
            lk_mod._LK_EXT = ext
            lk_mod._LK_EXT_TRIED = False

    import os
    loss_e, a_e, g_e = run("cuda", force_eager=True)
    loss_c, a_c, g_c = run("cuda", force_eager=False)
    assert abs(loss_e - loss_c) < 1e-4, f"[cuda {mode}] loss {loss_e} != {loss_c}"
    assert abs(a_e - a_c) < 1e-4, f"[cuda {mode}] alpha {a_e} != {a_c}"
    assert torch.allclose(g_e, g_c, rtol=1e-2, atol=1e-4), \
        f"[cuda {mode}] grad diff {(g_e - g_c).abs().max().item()}"
    src = "table" if use_table else "hiddens"
    print(f"  ok cuda-kernel == eager [{mode}/{src}]: loss={loss_c:.6f} "
          f"max|dg|={(g_e - g_c).abs().max().item():.2e}")


run_cuda_case("hybrid", use_table=True)
run_cuda_case("lk_alpha", use_table=True)
run_cuda_case("hybrid", use_table=False)
run_cuda_case("lk_alpha", use_table=False)

# degenerate case from the doc: point-mass target p -> LK-alpha == NLL of q.
# Identity head (C == V) so teacher hiddens ARE teacher logits and the point
# mass is exact.
Vp = 33
z = torch.randn(1, 4, Vp, requires_grad=True)
weight_id = torch.eye(Vp)
t_logits = torch.full((1, 4, Vp), -30.0)
t_logits[..., 7] = 30.0
loss_lk, a_lk, _, _ = _lk_loss(z, t_logits, weight_id,
                               torch.full((1, 4), 7), IGNORE,
                               "lk_alpha", 3.0, chunk=4)
nll = F.cross_entropy(z.reshape(-1, Vp), torch.full((4,), 7), reduction="mean")
assert torch.allclose(loss_lk, nll, rtol=1e-4, atol=1e-5), \
    f"point-mass degenerate case: lk {loss_lk.item()} vs nll {nll.item()}"
q7_mean = F.softmax(z.detach(), dim=-1)[..., 7].mean().item()
assert abs(a_lk.item() - q7_mean) < 1e-6
print(f"  ok point-mass p -> LK-alpha == NLL (lk={loss_lk.item():.6f}, "
      f"nll={nll.item():.6f}, alpha={a_lk.item():.6f})")

print("all test_lk_loss checks passed")
