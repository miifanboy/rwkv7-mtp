#!/usr/bin/env python3
"""Generate a binary numeric-parity fixture for tests/test-rwkv7-mtp.cpp.

The fixture is a plain little-endian binary blob:

    magic   "RWMT"      (4 bytes)
    int32   n_embd      C
    int32   n_vocab     V
    int32   n_cases     N
    per case:
        int32   token id
        f32     h[C]
        f32     z[C]              expected MTP head output (t_h_nextn)
        f32     logits[V]         expected output.weight @ z
        int32   argmax

Expected values are computed from the GGUF tensors (float16 weights) with
float32 numpy math -- exactly the arithmetic the C++ graph performs -- so the
C++ test catches norm-type / concat-order / missing-norm regressions.

Both MTP head flavors are supported (auto-detected from the GGUF tensors):

  * mlp   : fusion + residual MLP.  Stateless, so every case is computed
    independently.
  * rwkv7 : fusion + one full RWKV-7 block.  The block carries a recurrent
    state (WKV + token shift), and the C++ test decodes the cases sequentially
    in one context, so this script carries the same state across cases in the
    exact same order.  That mirrors the driver's prefill/catch-up decode.

Usage:
    python scripts/make_mtp_fixture.py --gguf mtp_out/mtp_heads.gguf --out mtp_out/mtp_fixture.bin
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
import struct

import numpy as np

import gguf

EPS = 1e-5
GN_EPS = 64e-5


def get(reader_tensors, name: str) -> np.ndarray:
    return np.asarray(reader_tensors[name].data, dtype=np.float32)


def layernorm(x, w, b):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + EPS) * w + b


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def detect_arch(T) -> str:
    # the rwkv7 head stores a full block under blk.0.nextn.time_mix_*; the mlp
    # head has no time-mix tensors at all
    return "rwkv7" if any(n.endswith(".nextn.time_mix_w0.weight") for n in T) else "mlp"


class Rwkv7Head:
    """Numpy emulation of the C++ graph_mtp RWKV-7 block path.

    Keeps the block's recurrent state (WKV `kv`, token-shift `x_prev`/`x2_prev`)
    so a sequence of single-token decodes matches llama.cpp exactly.
    """

    def __init__(self, T, C, head_size):
        g = lambda n: get(T, "blk.0.nextn." + n)  # noqa: E731
        self.g = g
        self.C = C
        self.N = head_size
        self.H = C // head_size

        self.hnorm_w = g("hnorm.weight")
        self.hnorm_b = g("hnorm.bias")
        self.enorm_w = g("enorm.weight")
        self.enorm_b = g("enorm.bias")
        self.eh_proj = g("eh_proj.weight")

        self.ln0_w, self.ln0_b = g("block_ln0.weight"), g("block_ln0.bias")
        self.ln1_w, self.ln1_b = g("attn_norm.weight"), g("attn_norm.bias")
        self.ln2_w, self.ln2_b = g("attn_norm_2.weight"), g("attn_norm_2.bias")

        self.lerp = g("time_mix_lerp_fused.weight").reshape(6, C)  # r,w,k,v,a,g
        self.w0 = g("time_mix_w0.weight")
        self.w1 = g("time_mix_w1.weight")
        self.w2 = g("time_mix_w2.weight")
        self.a0 = g("time_mix_a0.weight")
        self.a1 = g("time_mix_a1.weight")
        self.a2 = g("time_mix_a2.weight")
        self.g1 = g("time_mix_g1.weight")
        self.g2 = g("time_mix_g2.weight")
        self.k_k = g("time_mix_k_k.weight")
        self.k_a = g("time_mix_k_a.weight")
        self.r_k = g("time_mix_r_k.weight").reshape(self.H, self.N)

        self.rec = g("time_mix_receptance.weight")
        self.key = g("time_mix_key.weight")
        self.val = g("time_mix_value.weight")
        self.out = g("time_mix_output.weight")
        self.ln_x_w, self.ln_x_b = g("time_mix_ln.weight"), g("time_mix_ln.bias")

        self.cm_lerp = g("channel_mix_lerp_k.weight")
        self.cm_key = g("channel_mix_key.weight")
        self.cm_val = g("channel_mix_value.weight")

        # recurrent state, zero-initialized like a fresh draft context
        self.kv = np.zeros((self.H, self.N, self.N), dtype=np.float32)
        self.x_prev = np.zeros(C, dtype=np.float32)    # previous row's att_norm
        self.x2_prev = np.zeros(C, dtype=np.float32)   # previous row's ffn_norm

    def step(self, h, tok_e):
        C, H, N = self.C, self.H, self.N

        # fusion: [ln_h(h), ln_tok(tok)] -- hidden FIRST
        x = np.concatenate([layernorm(h, self.hnorm_w, self.hnorm_b),
                            layernorm(tok_e, self.enorm_w, self.enorm_b)]) @ self.eh_proj.T

        # Block(layer_id=0): x = ln0(x) first
        base = layernorm(x, self.ln0_w, self.ln0_b)
        xn = layernorm(base, self.ln1_w, self.ln1_b)

        d = self.x_prev - xn
        xr = xn + d * self.lerp[0]
        xw = xn + d * self.lerp[1]
        xk = xn + d * self.lerp[2]
        xv = xn + d * self.lerp[3]
        xa = xn + d * self.lerp[4]
        xg = xn + d * self.lerp[5]

        r = xr @ self.rec.T
        # the graph form exp(-0.606531*sigmoid(w0 + w_m)) (== exp(-exp(w_pre)))
        wm = np.tanh(xw @ self.w1.T) @ self.w2.T
        w_dec = np.exp(sigmoid(self.w0 + wm) * -0.606531)
        k = xk @ self.key.T
        v = xv @ self.val.T
        a = sigmoid(self.a0 + (xa @ self.a1.T) @ self.a2.T)
        g = sigmoid(xg @ self.g1.T) @ self.g2.T

        kk3 = (k * self.k_k).reshape(H, N)
        kk3 = kk3 / np.maximum(np.linalg.norm(kk3, axis=-1, keepdims=True), 1e-12)
        k2 = k * (1 + (a - 1) * self.k_a)

        # one wkv7 step (ggml_compute_forward_rwkv_wkv7_f32, T=1).
        # The kernel is invoked as ggml_rwkv_wkv7(r, w, k, v, -kk, kk*a, st)
        # (see RWKV_Tmix_x070: ``RWKV7_CLAMPW_CUDA(r, w, k, v, -kk, kk*a)``,
        # spec_decode.RWKV7HeadRunner._run, and the llama.cpp graph), i.e.
        #   sa = (-kk) . st ;  st' = st*w + v (x) k + sa * (kk*a)
        r3, w3 = r.reshape(H, N), w_dec.reshape(H, N)
        k3, v3, a3 = k2.reshape(H, N), v.reshape(H, N), a.reshape(H, N)
        b3 = kk3 * a3
        sa = np.einsum("hj,hij->hi", -kk3, self.kv)
        self.kv = (self.kv * w3[:, None, :]
                   + np.einsum("hi,hj->hij", v3, k3)
                   + sa[:, :, None] * b3[:, None, :])
        y = np.einsum("hij,hj->hi", self.kv, r3)             # (H, N)

        # group norm per (head) over head_size (displayed as (N, H, T) in ggml),
        # then per-channel scale/shift
        y3 = y.reshape(1, H, N)
        y3 = (y3 - y3.mean(-1, keepdims=True)) / np.sqrt(y3.var(-1, keepdims=True) + GN_EPS)
        y = y3.reshape(C) * self.ln_x_w + self.ln_x_b

        rk = (r3 * k3 * self.r_k).sum(-1, keepdims=True)     # (H, 1)
        y = (y.reshape(H, N) + rk * v3).reshape(C)
        att_out = (y * g) @ self.out.T

        ffn_inp = base + att_out
        xn2 = layernorm(ffn_inp, self.ln2_w, self.ln2_b)
        k2f = xn2 + (self.x2_prev - xn2) * self.cm_lerp
        k2f = np.maximum(k2f @ self.cm_key.T, 0.0) ** 2
        z = (k2f @ self.cm_val.T) + ffn_inp

        self.x_prev = xn
        self.x2_prev = xn2
        return z


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", type=int, default=8)
    args = ap.parse_args()

    r = gguf.GGUFReader(args.gguf)
    T = {t.name: t for t in r.tensors}

    C = int(r.fields["rwkv7.embedding_length"].contents())
    V = len(r.fields["tokenizer.ggml.tokens"].data)
    K = int(r.fields["rwkv7.nextn_predict_layers"].contents())
    assert K >= 1, K

    arch = detect_arch(T)
    head_size = int(r.fields["rwkv7.wkv.head_size"].contents())

    tok_embd = get(T, "token_embd.weight")
    head = get(T, "output.weight")

    # fusion tensors (shared by both head layouts)
    hnorm_w = get(T, "blk.0.nextn.hnorm.weight")
    hnorm_b = get(T, "blk.0.nextn.hnorm.bias")
    enorm_w = get(T, "blk.0.nextn.enorm.weight")
    enorm_b = get(T, "blk.0.nextn.enorm.bias")
    eh_proj = get(T, "blk.0.nextn.eh_proj.weight")

    if arch == "mlp":
        mlp_norm_w = get(T, "blk.0.nextn.mlp_norm.weight") if "blk.0.nextn.mlp_norm.weight" in T else None
        mlp_norm_b = get(T, "blk.0.nextn.mlp_norm.bias") if "blk.0.nextn.mlp_norm.bias" in T else None
        mlp_up = get(T, "blk.0.nextn.mlp_up.weight") if "blk.0.nextn.mlp_up.weight" in T else None
        mlp_down = get(T, "blk.0.nextn.mlp_down.weight") if "blk.0.nextn.mlp_down.weight" in T else None
    else:
        block = Rwkv7Head(T, C, head_size)

    rng = np.random.default_rng(1234)

    cases = []
    for i in range(args.cases):
        tok = int(rng.integers(0, V))
        # random post-output_norm-scale hidden (RWKV hidden states are O(1)ish)
        h = (rng.standard_normal(C).astype(np.float32) * 0.5)

        e = tok_embd[tok]
        hn = layernorm(h, hnorm_w, hnorm_b)
        en = layernorm(e, enorm_w, enorm_b)
        concat = np.concatenate([hn, en], axis=-1)  # [hidden, token] -- hidden FIRST

        if arch == "mlp":
            z0 = concat @ eh_proj.T
            if mlp_up is not None:
                mn = layernorm(z0, mlp_norm_w, mlp_norm_b)
                z = z0 + silu(mn @ mlp_up.T) @ mlp_down.T
            else:
                z = z0
        else:
            # stateful: carries the block state across the case sequence, in the
            # same order the C++ test decodes them
            z = block.step(h, e)

        assert z.shape == (C,), z.shape
        logits = z @ head.T
        cases.append((tok, h, z.astype(np.float32), logits.astype(np.float32), int(logits.argmax())))

    with open(args.out, "wb") as f:
        f.write(b"RWMT")
        f.write(struct.pack("<iii", C, V, len(cases)))
        for tok, h, z, logits, am in cases:
            f.write(struct.pack("<i", tok))
            f.write(h.astype("<f4").tobytes())
            f.write(z.astype("<f4").tobytes())
            f.write(logits.astype("<f4").tobytes())
            f.write(struct.pack("<i", am))

    print(f"[fixture] wrote {args.out}: arch={arch} C={C} V={V} cases={len(cases)}")


if __name__ == "__main__":
    main()
