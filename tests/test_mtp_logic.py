"""
CPU-only unit tests for the MTP math.  These do NOT import the RWKV backbone
(and therefore do not touch CUDA), so they run on any machine:

    python tests/test_mtp_logic.py

The RWKV-7 block head (``--mtp-arch rwkv7``) needs the CUDA kernels; its tests
live in ``test_rwkv7_head.py`` (CUDA-gated, skips on CPU-only machines).
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
import torch.nn as nn

from mtp import (MTPModule, mtp_loss, run_shift_selftest, _shift_left, _ce,
                 _head_ce, CE_CHUNK)


def test_shift():
    run_shift_selftest()
    print("PASS  test_shift")


def test_mtp_module_shapes():
    B, T, C = 2, 8, 64
    m = MTPModule(C, hidden_mult=2.0)
    h = torch.randn(B, T, C)
    tok = torch.randn(B, T, C)
    z = m(h, tok)
    assert z.shape == (B, T, C), z.shape
    m_nomlp = MTPModule(C, use_mlp=False)
    assert m_nomlp(h, tok).shape == (B, T, C)
    print(f"PASS  test_mtp_module_shapes  ({sum(p.numel() for p in m.parameters()):,} params/head)")


def _perfect_logits(targets, V):
    logits = torch.full((*targets.shape, V), -10.0)
    logits.scatter_(-1, targets.unsqueeze(-1), 20.0)
    return logits


def test_mtp_loss_perfect():
    torch.manual_seed(0)
    B, T, V = 2, 8, 32
    targets = torch.randint(1, V, (B, T))
    main = _perfect_logits(targets, V)
    mtp_logits = [_perfect_logits(_shift_left(targets, k + 1, pad=0), V) for k in range(3)]
    loss, parts = mtp_loss(main, mtp_logits, targets, decay=0.8, ignore_index=0,
                           return_parts=True)
    assert loss.item() < 0.1, loss.item()
    for key in ("main", "mtp_1", "mtp_2", "mtp_3", "total"):
        assert key in parts
    print(f"PASS  test_mtp_loss_perfect  (loss={loss.item():.4f})")


def test_ignore_index_masking():
    """Positions whose target is pad must not contribute to the loss."""
    torch.manual_seed(0)
    B, T, V = 2, 6, 16
    targets = torch.randint(1, V, (B, T))
    targets[:, -1] = 0  # pad
    logits = torch.randn(B, T, V)

    loss_masked = mtp_loss(logits, [], targets, ignore_index=0).item()

    import torch.nn.functional as F
    flat_logits = logits.reshape(-1, V)
    flat_t = targets.reshape(-1)
    valid = flat_t != 0
    manual = F.cross_entropy(flat_logits[valid], flat_t[valid]).item()
    assert abs(loss_masked - manual) < 1e-6, (loss_masked, manual)
    print(f"PASS  test_ignore_index_masking  ({loss_masked:.4f})")


def test_grad_flow_and_freeze():
    """Chained heads train; the frozen embedding and shared head get no grad."""
    torch.manual_seed(0)
    B, T, C, V = 2, 16, 32, 53
    num_mtp = 3

    emb = nn.Embedding(V, C)
    head = nn.Linear(C, V, bias=False)
    for p in list(emb.parameters()) + list(head.parameters()):
        p.requires_grad_(False)

    heads = nn.ModuleList([MTPModule(C) for _ in range(num_mtp)])
    # Detached stand-in for the frozen backbone hidden state.
    hidden = torch.randn(B, T, C)

    idx = torch.randint(1, V, (B, T))
    targets = _shift_left(idx, 1, pad=0)

    opt = torch.optim.AdamW(heads.parameters(), lr=3e-3)

    def step_loss():
        h_prev = hidden
        mtp_logits = []
        for k, m in enumerate(heads):
            tok = emb(_shift_left(idx, k + 1, pad=0))
            z = m(h_prev, tok)
            mtp_logits.append(head(z))
            h_prev = z
        return mtp_loss(head(hidden), mtp_logits, targets, decay=0.8, ignore_index=0)

    first = step_loss().item()
    for _ in range(80):
        opt.zero_grad()
        loss = step_loss()
        loss.backward()

        # frozen modules must not accumulate grads
        assert all(p.grad is None for p in head.parameters()), "shared head is not frozen"
        assert all(p.grad is None for p in emb.parameters()), "embedding is not frozen"
        # every MTP module must receive gradient
        for k, m in enumerate(heads):
            assert any(p.grad is not None for p in m.parameters()), f"depth {k} got no grad"

        opt.step()

    last = step_loss().item()
    assert last < 0.6 * first, f"loss did not drop enough: {first:.4f} -> {last:.4f}"
    print(f"PASS  test_grad_flow_and_freeze  (loss {first:.4f} -> {last:.4f})")


def test_memory_efficient_loss_matches_reference():
    """forward_loss-style per-depth CE must equal mtp_loss on the same logits."""
    torch.manual_seed(0)
    B, T, V = 2, 10, 24
    targets = torch.randint(1, V, (B, T))
    logits = [torch.randn(B, T, V) for _ in range(4)]
    ref = mtp_loss(logits[0], logits[1:], targets, decay=0.8, ignore_index=0).item()

    # replicate the one-depth-at-a-time reduction
    ce_main = _ce(logits[0], targets, 0)
    loss = ce_main
    total_w = 1.0
    for k, lg in enumerate(logits[1:]):
        ce_k = _ce(lg, _shift_left(targets, k + 1, pad=0), 0)
        w = 0.8 ** (k + 1)
        loss = loss + w * ce_k
        total_w += w
    manual = (loss / total_w).item()
    assert abs(ref - manual) < 1e-5, (ref, manual)
    print(f"PASS  test_memory_efficient_loss_matches_reference  ({ref:.6f})")


def test_ce_chunk_invariance():
    """Chunk size must not change the loss, with and without ignore_index."""
    torch.manual_seed(0)
    B, T, V = 2, 37, 24          # 74 rows: exercises partial final chunks
    logits = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))
    targets[0, ::3] = 0          # sprinkle ignore_index

    ref = _ce(logits, targets, 0, chunk=10 ** 9).item()
    for chunk in (1, 5, 16, 64, 10 ** 9):
        got = _ce(logits, targets, 0, chunk=chunk).item()
        assert abs(got - ref) < 1e-5, (chunk, got, ref)

    # cross-check against a hand-rolled masked mean
    import torch.nn.functional as F
    flat, tgt = logits.reshape(-1, V), targets.reshape(-1)
    valid = tgt != 0
    manual = F.cross_entropy(flat[valid], tgt[valid]).item()
    assert abs(ref - manual) < 1e-6, (ref, manual)
    print(f"PASS  test_ce_chunk_invariance  (default chunk={CE_CHUNK})")


def test_fused_head_ce_matches_reference():
    """_head_ce must match head(z) + _ce in both value and input gradient."""
    torch.manual_seed(0)
    B, T, C, V = 2, 12, 16, 29
    z = torch.randn(B, T, C)
    weight = torch.randn(V, C)
    targets = torch.randint(0, V, (B, T))
    targets[0, -1] = 0                      # exercise ignore_index
    targets[1, :2] = 0

    z_ref = z.clone().requires_grad_(True)
    z_new = z.clone().requires_grad_(True)
    ref = _ce(z_ref @ weight.t(), targets, ignore_index=0)
    new = _head_ce(z_new, weight, targets, ignore_index=0, chunk=5)
    assert abs(ref.item() - new.item()) < 1e-5, (ref.item(), new.item())

    ref.backward()
    new.backward()
    assert z_new.grad is not None
    assert torch.allclose(z_ref.grad, z_new.grad, atol=1e-5), \
        (z_ref.grad - z_new.grad).abs().max().item()
    print("PASS  test_fused_head_ce_matches_reference  (value + dz)")


def test_fused_head_ce_weight_grad_and_freeze():
    """Frozen head -> no grad (and nothing retained); trainable head -> matches."""
    torch.manual_seed(1)
    B, T, C, V = 2, 6, 8, 13
    z = torch.randn(B, T, C)
    weight = torch.randn(V, C)
    targets = torch.randint(0, V, (B, T))

    # frozen weight: backward must not produce a weight gradient
    z_g = z.clone().requires_grad_(True)
    w_frozen = weight.clone().requires_grad_(False)
    _head_ce(z_g, w_frozen, targets, ignore_index=0, chunk=4).backward()
    assert w_frozen.grad is None, "frozen head accumulated a gradient"
    assert z_g.grad is not None

    # trainable weight: gradient must match the reference path
    w_ref = weight.clone().requires_grad_(True)
    w_new = weight.clone().requires_grad_(True)
    _ce(z @ w_ref.t(), targets, ignore_index=0).backward()
    _head_ce(z, w_new, targets, ignore_index=0, chunk=4).backward()
    assert torch.allclose(w_ref.grad, w_new.grad, atol=1e-5), \
        (w_ref.grad - w_new.grad).abs().max().item()
    print("PASS  test_fused_head_ce_weight_grad_and_freeze")


def test_fused_head_ce_end_to_end_grad_flow():
    """Mirror of forward_loss: only MTP params train, head/emb stay grad-free."""
    torch.manual_seed(0)
    B, T, C, V = 2, 16, 32, 53
    emb = nn.Embedding(V, C)
    head = nn.Linear(C, V, bias=False)
    for p in list(emb.parameters()) + list(head.parameters()):
        p.requires_grad_(False)

    heads = nn.ModuleList([MTPModule(C) for _ in range(3)])
    hidden = torch.randn(B, T, C)
    idx = torch.randint(1, V, (B, T))
    targets = _shift_left(idx, 1, pad=0)

    h_prev, loss, total_w = hidden, _head_ce(hidden, head.weight, targets, 0), 1.0
    for k, m in enumerate(heads):
        tok = emb(_shift_left(idx, k + 1, pad=0))
        z = m(h_prev, tok)
        ce = _head_ce(z, head.weight, _shift_left(targets, k + 1, pad=0), 0)
        w = 0.8 ** (k + 1)
        loss = loss + w * ce
        total_w += w
        h_prev = z
    (loss / total_w).backward()

    assert all(p.grad is None for p in head.parameters()), "shared head got grad"
    assert all(p.grad is None for p in emb.parameters()), "embedding got grad"
    for k, m in enumerate(heads):
        assert any(p.grad is not None for p in m.parameters()), f"depth {k} got no grad"
    print("PASS  test_fused_head_ce_end_to_end_grad_flow")


def test_mtp_arch_checkpoint_roundtrip():
    """save_mtp_heads records the head arch; load_mtp_heads enforces it."""
    import os
    import tempfile
    import types

    from mtp import load_mtp_heads, peek_mtp_arch, save_mtp_heads

    def stub(arch):
        m = nn.Module()
        m.args = types.SimpleNamespace(n_embd=8)
        m.num_mtp = 2
        m.mtp_arch = arch
        m.mtp_heads = nn.ModuleList([nn.Linear(8, 8, bias=False)])
        return m

    with tempfile.TemporaryDirectory() as td:
        p_mlp = os.path.join(td, "mlp.pt")
        p_rk = os.path.join(td, "rwkv7.pt")
        save_mtp_heads(stub("mlp"), p_mlp)
        save_mtp_heads(stub("rwkv7"), p_rk)
        assert peek_mtp_arch(p_mlp) == "mlp"
        assert peek_mtp_arch(p_rk) == "rwkv7"
        meta = load_mtp_heads(stub("mlp"), p_mlp)   # matching arch loads fine
        assert meta["arch"] == "mlp"
        try:
            load_mtp_heads(stub("mlp"), p_rk)
            raise AssertionError("arch mismatch was not caught")
        except RuntimeError as e:
            assert "--mtp-arch" in str(e), e
    print("PASS  test_mtp_arch_checkpoint_roundtrip")


if __name__ == "__main__":
    test_shift()
    test_mtp_module_shapes()
    test_mtp_loss_perfect()
    test_ignore_index_masking()
    test_memory_efficient_loss_matches_reference()
    test_grad_flow_and_freeze()
    test_ce_chunk_invariance()
    test_fused_head_ce_matches_reference()
    test_fused_head_ce_weight_grad_and_freeze()
    test_fused_head_ce_end_to_end_grad_flow()
    test_mtp_arch_checkpoint_roundtrip()
    print("\nAll MTP logic tests passed.")
