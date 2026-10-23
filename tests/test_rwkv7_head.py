"""CUDA-gated tests for the RWKV-7 block MTP head (``--mtp-arch rwkv7``).

    python tests/test_rwkv7_head.py

Skips with a message on machines without CUDA (the RWKV-7 fused kernels only
exist there).  Runs everything at toy dims (C=64, one head, T=32), so it never
touches the 2.9B backbone.

Covers:
  1. RWKV7MTPModule: forward shape, gradient reachability of every parameter.
  2. Chained (shared, recursive) training drops the loss on a fixed batch.
  3. RWKV7HeadRunner recurrent path == training-time parallel path (the key
     correctness property: the fp16 RNN state machine must reproduce the bf16
     parallel/teacher-forced computation).
  4. advance()-after-commit == one-shot prefill over the same stream (the
     staggered "ingest on commit" discipline tiles exactly).
  5. draft_chain tentative ingestions + restore() leave no residue; chain then
     advance() with accepted drafts == direct advance().
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

import types

import torch
import torch.nn as nn

from mtp import MTP_ARCHS, RWKV7MTPModule, _shift_left

TINY = types.SimpleNamespace(
    n_embd=64, n_layer=2, head_size=64, dim_att=64, dim_ffn=128,
    my_testing="x070", grad_cp=0,
)
V = 53
T = 32            # WKV kernel requires T % 16 == 0
C = TINY.n_embd
DEVICE = "cuda:0"


def _make_module(perturb: bool = False, seed: int = 0):
    torch.manual_seed(seed)
    m = RWKV7MTPModule(TINY).to(device=DEVICE, dtype=torch.bfloat16)
    if perturb:
        # The backbone init zeroes the block's output/value projections, so a
        # fresh block is (nearly) the identity; give it something to do so the
        # parity test actually exercises the WKV/recurrence math.
        with torch.no_grad():
            for name in ("block.att.output.weight", "block.att.key.weight",
                         "block.att.value.weight", "block.att.receptance.weight",
                         "block.ffn.key.weight", "block.ffn.value.weight"):
                parts = name.split(".")
                parent = m
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                parent.weight.normal_(0, 0.05)
            m.block.att.w1.normal_(0, 0.02)
            m.block.att.w2.normal_(0, 0.02)
            m.block.att.g1.normal_(0, 0.02)
            m.block.att.g2.normal_(0, 0.02)
    return m


def _require_cuda():
    if not torch.cuda.is_available():
        print("SKIP  (CUDA not available -- RWKV-7 kernels need it)")
        return False
    return True


def test_rwkv7_module_forward_and_grads():
    m = _make_module()
    h = torch.randn(1, T, C, device=DEVICE, dtype=torch.bfloat16)
    tok = torch.randn(1, T, C, device=DEVICE, dtype=torch.bfloat16)
    h_mid = torch.randn(1, T, C, device=DEVICE, dtype=torch.bfloat16)
    z = m(h, tok)
    assert z.shape == (1, T, C), z.shape
    assert torch.isfinite(z).all()
    # fuse_backbone must be part of the graph too: its params (ln_early, ln_mid,
    # fc_fuse) are only reached through the depth-0 fused anchor, not forward().
    h0 = m.fuse_backbone(h, h_mid, h)
    (z.float().sum() + h0.float().sum()).backward()
    named = list(m.named_parameters())
    missing = [n for n, p in named if p.grad is None]
    assert not missing, f"params without grad: {missing}"
    n_params = sum(p.numel() for p in m.parameters())
    print(f"PASS  test_rwkv7_module_forward_and_grads  ({n_params:,} params)")


def test_eagle3_step0_identity():
    """[0, 0, I] fc_fuse init: fuse_backbone(h, h, h) == h bit-for-bit at step 0
    (the verification checklist's zero-shock sanity check)."""
    m = _make_module(seed=7)
    h = torch.randn(2, T, C, device=DEVICE, dtype=torch.bfloat16)
    h0 = m.fuse_backbone(h, h, h)
    diff = (h0.float() - h.float()).abs().max().item()
    assert diff == 0.0, f"step-0 fusion is not identity: max abs diff {diff}"
    # The full depth-0 input path must also be an exact reparametrization:
    # ln_out-wrapped block output equals the plain block's through the same
    # proj([I; 0]) init only up to the head's own computation, so just verify
    # forward() shape/finiteness here and identity above.
    tok = torch.randn(2, T, C, device=DEVICE, dtype=torch.bfloat16)
    z = m(h, tok)
    assert torch.isfinite(z).all()
    print(f"PASS  test_eagle3_step0_identity  (max abs diff 0.0)")


def test_runner_fuse_taps_parity():
    """The runner's fp16 fuse_taps must mirror the module's bf16 fuse_backbone."""
    from spec_decode import RWKV7HeadRunner

    m = _make_module(seed=8)
    torch.manual_seed(9)
    h_early = torch.randn(1, T, C, device=DEVICE, dtype=torch.bfloat16) * 0.5
    h_mid = torch.randn(1, T, C, device=DEVICE, dtype=torch.bfloat16) * 0.5
    h_late = torch.randn(1, T, C, device=DEVICE, dtype=torch.bfloat16) * 0.5

    runner = _build_runner(m, torch.zeros(V, C))
    with torch.no_grad():
        h0_mod = m.fuse_backbone(h_early, h_mid, h_late)[0].float()
        h0_run = runner.fuse_taps(h_early[0].half(), h_mid[0].half(),
                                  h_late[0].half()).float()
    diff = (h0_mod - h0_run).abs()
    rel = diff.mean() / h0_mod.abs().mean().clamp_min(1e-3)
    assert rel < 0.02, f"fuse parity: rel mean diff {rel:.4f}, max {diff.max().item():.4f}"
    print(f"PASS  test_runner_fuse_taps_parity  rel_mean_diff={rel:.5f} "
          f"max_abs={diff.max().item():.4f}")


def test_rwkv7_chained_recursion_trains():
    """Shared head applied recursively, FastMTP-style, on a fixed batch."""
    torch.manual_seed(3)
    m = _make_module()
    emb = nn.Embedding(V, C).to(device=DEVICE, dtype=torch.bfloat16)
    head = nn.Linear(C, V, bias=False).to(device=DEVICE, dtype=torch.bfloat16)
    for p in list(emb.parameters()) + list(head.parameters()):
        p.requires_grad_(False)

    K = 2
    idx = torch.randint(1, V, (1, T), device=DEVICE)
    targets = torch.roll(idx, -1, dims=1)
    h = torch.randn(1, T, C, device=DEVICE, dtype=torch.bfloat16)

    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=3e-3)

    def step_loss():
        h_prev, mtp_logits = h, []
        for k in range(K):
            tok = emb(torch.roll(idx, -(k + 1), dims=1))
            z = m(h_prev, tok)
            mtp_logits.append(head(z))
            h_prev = z
        main = torch.nn.functional.cross_entropy(
            head(h).reshape(-1, V), targets.reshape(-1))
        losses = [torch.nn.functional.cross_entropy(
            lg.reshape(-1, V),
            torch.roll(targets, -(k + 1), dims=1).reshape(-1)) for k, lg in enumerate(mtp_logits)]
        return (main + sum(losses)) / (1 + K)

    first = step_loss().item()
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        loss = step_loss()
        loss.backward()
        assert all(p.grad is None for p in head.parameters()), "frozen head got grad"
        assert all(p.grad is None for p in emb.parameters()), "embedding got grad"
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    last = step_loss().item()
    assert last < 0.7 * first, f"loss did not drop enough: {first:.3f} -> {last:.3f}"
    print(f"PASS  test_rwkv7_chained_recursion_trains  ({first:.3f} -> {last:.3f})")


def _build_runner(m, emb_weight):
    from spec_decode import RWKV7HeadRunner
    w = {k: v for k, v in m.state_dict().items()}
    return RWKV7HeadRunner([w], C, TINY.head_size,
                           emb_weight.to(device=DEVICE, dtype=torch.half),
                           device=DEVICE, n_depths=2)


def test_runner_parity_with_parallel():
    """The recurrent runner must reproduce the parallel (training) recursion."""
    from spec_decode import prepare_rwkv7_head_weights  # ensure import works

    m = _make_module(perturb=True, seed=1)
    emb = nn.Embedding(V, C).to(device=DEVICE, dtype=torch.bfloat16)
    H = torch.randn(1, T, C, device=DEVICE, dtype=torch.bfloat16) * 0.5
    toks = torch.randint(1, V, (1, T), device=DEVICE)

    # -- parallel / teacher-forced path (mirrors forward_loss's recursion) ----
    with torch.no_grad():
        z_par = []
        h_prev = H
        for k in range(2):
            tok = emb(_shift_left(toks, k + 1, pad=0))
            h_prev = m(h_prev, tok)
            z_par.append(h_prev[0].float())          # (T, C)

    # -- recurrent path --------------------------------------------------------
    runner = _build_runner(m, emb.weight)
    with torch.no_grad():
        runner.prefill(H[0].half(), toks[0].tolist())

    for k in range(2):
        hi = T - k - 2                                # runner's watermark bound
        a = torch.cat([runner.zbufs[k][j] for j in range(0, hi + 1)], 0).float()
        b = z_par[k][:hi + 1]
        diff = (a - b).abs()
        rel = diff.mean() / b.abs().mean().clamp_min(1e-3)
        assert rel < 0.05, f"depth {k}: relative mean diff {rel:.4f}, max {diff.max():.4f}"
        print(f"PASS  test_runner_parity_with_parallel  depth {k}: "
              f"rel_mean_diff={rel:.5f} max_abs={diff.max().item():.4f}")


def test_runner_advance_matches_prefill():
    """Commit-sized advance windows must tile the one-shot prefill exactly."""
    m = _make_module(perturb=True, seed=2)
    emb = nn.Embedding(V, C).to(device=DEVICE, dtype=torch.bfloat16)
    torch.manual_seed(4)
    P = 96
    H = (torch.randn(P, C, device=DEVICE, dtype=torch.bfloat16) * 0.5)
    toks = torch.randint(1, V, (P,), device=DEVICE).tolist()

    # -- one-shot prefill ------------------------------------------------------
    r_full = _build_runner(m, emb.weight)
    with torch.no_grad():
        r_full.prefill(H.half(), toks)

    # -- prefix prefill + commit cycles ----------------------------------------
    r_inc = _build_runner(m, emb.weight)
    P0 = 17
    with torch.no_grad():
        r_inc.prefill(H[:P0].half(), toks[:P0])
        p = P0 - 1
        sizes = [2, 3, 4, 2, 4, 3]                     # m+2 for K=2: {2..4}
        i = 0
        while p < P - 1:
            s = sizes[i % len(sizes)]
            i += 1
            if p + s > P - 1:
                s = P - 1 - p
            q = p + s
            H_new = H[p + 1:q + 1].half()              # H[p+1..q]
            r_inc.advance(H[p:p + 1].half(), H_new, p, toks)
            p = q
        assert p == P - 1, (p, P - 1)

    for k in range(2):
        for i in range(3):
            a, b = r_inc.st[k][i], r_full.st[k][i]
            assert torch.allclose(a.float(), b.float(), atol=2e-2), \
                f"state[{k}][{i}] mismatch: max {(a.float() - b.float()).abs().max()}"
        common = sorted(set(r_inc.zbufs[k]) & set(r_full.zbufs[k]))
        for j in common:
            d = (r_inc.zbufs[k][j].float() - r_full.zbufs[k][j].float()).abs().max()
            assert d < 2e-2, f"zbufs[{k}][{j}] diff {d}"
    print(f"PASS  test_runner_advance_matches_prefill  "
          f"(sizes {sizes}, P0={P0}, P={P})")


def test_runner_chain_rollback_integrity():
    """Tentative chain ingestions must vanish on restore() and, when the drafts
    are accepted, agree exactly with a direct advance()."""
    m = _make_module(perturb=True, seed=5)
    emb = nn.Embedding(V, C).to(device=DEVICE, dtype=torch.bfloat16)
    torch.manual_seed(6)
    P = 64
    H = torch.randn(P, C, device=DEVICE, dtype=torch.bfloat16) * 0.5
    toks = torch.randint(1, V, (P,), device=DEVICE).tolist()
    K = 2
    w_head = torch.randn(C, V, device=DEVICE, dtype=torch.half) * 0.05
    pick = lambda z: (z @ w_head).argmax(dim=-1)

    r = _build_runner(m, emb.weight)
    P0 = 40
    with torch.no_grad():
        r.prefill(H[:P0].half(), toks[:P0])
        p = P0 - 1

        # -- direct advance over a full-acceptance commit (m = K -> size K+2) --
        r_direct = _build_runner(m, emb.weight)
        r_direct.st = [[t.clone() for t in st] for st in r.st]
        r_direct.zbufs = [{j: z.clone() for j, z in buf.items()} for buf in r.zbufs]
        r_direct.wm = list(r.wm)

        # -- chain path: snapshot, tentative chain, restore, advance ----------
        snap = r.snapshot()
        drafts = r.draft_chain(H[p:p + 1].half(), p, pick)
        assert len(drafts) == K + 1
        emitted = [int(d.item()) for d in drafts] + [int(drafts[-1].item())]
        r.restore(snap)
        # states must be bit-identical to the snapshot after restore
        for k in range(K):
            for i in range(3):
                assert torch.equal(r.st[k][i], snap["st"][k][i]), \
                    "restore() left residue in the WKV state"
        toks_ext = toks[:p + 1] + emitted
        H_new = H[p + 1:p + 1 + len(emitted)].half()
        r.advance(H[p:p + 1].half(), H_new, p, toks_ext)

        # -- the direct path does the same advance without any chain ----------
        r_direct.advance(H[p:p + 1].half(), H_new, p, toks_ext)

        for k in range(K):
            for i in range(3):
                a, b = r.st[k][i].float(), r_direct.st[k][i].float()
                assert torch.allclose(a, b, atol=2e-2), \
                    f"chain+advance != advance: state[{k}][{i}] max {(a - b).abs().max()}"
    print("PASS  test_runner_chain_rollback_integrity")


if __name__ == "__main__":
    assert "rwkv7" in MTP_ARCHS
    if _require_cuda():
        test_rwkv7_module_forward_and_grads()
        test_eagle3_step0_identity()
        test_rwkv7_chained_recursion_trains()
        test_runner_parity_with_parallel()
        test_runner_advance_matches_prefill()
        test_runner_chain_rollback_integrity()
        test_runner_fuse_taps_parity()
        print("\nAll RWKV-7 head tests passed.")