"""Verify mtp_heads.gguf: metadata, tensor map, and a numerical round-trip of
the MTP head forward pass computed purely from the GGUF tensors.

Supports both --mtp-arch flavors:
  * mlp   : fusion + residual MLP (stateless).
  * rwkv7 : fusion + one full RWKV-7 block (stateful).  The GGUF-side emulation
    reproduces the llama.cpp draft graph exactly (including the recurrent WKV
    state, token shift and the exp(-0.606531*sigmoid(w)) decay transform); the
    reference is a pure-pytorch port of the trained head math (mtp.RWKV7MTPModule
    / spec_decode.RWKV7HeadRunner._run with the wkv7 recurrence unrolled).
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

import sys

import numpy as np
import torch

import gguf

GGUF = sys.argv[1] if len(sys.argv) > 1 else "mtp_out/mtp_heads.gguf"
PT = sys.argv[2] if len(sys.argv) > 2 else "mtp_out/mtp_heads.pt"
PTH = sys.argv[3] if len(sys.argv) > 3 else "rwkv7-g1k-2.9b-20260930-ctx25600.pth"

r = gguf.GGUFReader(GGUF)

print("=== metadata ===")
for f in r.fields.values():
    if f.name.startswith(("general.", "rwkv7.", "tokenizer.ggml.model",
                          "tokenizer.ggml.bos", "tokenizer.ggml.eos",
                          "tokenizer.ggml.eot", "tokenizer.ggml.tokens")):
        if f.name == "tokenizer.ggml.tokens":
            print(f"  {f.name:42} <{len(f.data)} entries>")
        else:
            print(f"  {f.name:42} {f.data if not hasattr(f.data, 'tolist') else f.data.tolist()[:4]}")

T = {}
for t in r.tensors:
    T[t.name] = t

# --- arch guard --------------------------------------------------------------
_pt_payload = torch.load(PT, map_location="cpu", weights_only=True)
_pt_arch = str(_pt_payload.get("arch")
               or (_pt_payload.get("meta") or {}).get("arch") or "mlp")
if _pt_arch not in ("mlp", "rwkv7"):
    raise SystemExit(
        f"{PT} declares --mtp-arch {_pt_arch!r}; verify_mtp_gguf supports "
        f"'mlp' and 'rwkv7'")
print(f"\ncheckpoint arch: {_pt_arch}")

print("\n=== tensors ===")
for name in sorted(T):
    t = T[name]
    print(f"  {name:38} {str(tuple(t.shape)):16} {t.tensor_type.name}")

# --- llama.cpp's auto-detect probe ------------------------------------------
bc = int(r.fields["rwkv7.block_count"].contents())
nn = int(r.fields["rwkv7.nextn_predict_layers"].contents())
probe = f"blk.{bc - 1}.nextn.eh_proj.weight"
print(f"\nblock_count={bc} nextn_predict_layers={nn} -> n_layer()={bc - nn}")
print(f"auto-detect probe {probe!r}: {'FOUND' if probe in T else 'MISSING'}")

# --- shared-head check (FastMTP style) ---------------------------------------
# An --mtp-arch mlp shared-head file replicates ONE head into blk.0..blk.{K-1};
# every block must hold bit-identical tensors.  A non-shared (per-depth) file
# must NOT.  An rwkv7 file always carries exactly one block (stateful heads are
# driven through the single-head recursion), so nothing to check there.
_shared_names = [n for n in T if n.startswith("blk.") and ".nextn." in n]
shared_file = bc > 1 and _shared_names and all(
    np.array_equal(np.asarray(T[n].data),
                   np.asarray(T[n.replace("blk." + n.split(".")[1], "blk.0")].data))
    for n in _shared_names if not n.startswith("blk.0.")
)
print(f"shared-head file (all {bc} blocks identical): {shared_file}")
try:
    pt_shared = bool(torch.load(PT, map_location="cpu", weights_only=True).get("shared_head", False))
    if _pt_arch == "rwkv7":
        if bc != 1:
            print("  WARNING: rwkv7 export must carry exactly one nextn block")
    elif pt_shared != shared_file:
        print(f"  WARNING: .pt says shared_head={pt_shared} but the GGUF looks "
              f"shared={shared_file} -- converter/checkpoint mismatch")
except Exception as e:
    print(f"  (could not cross-check .pt shared flag: {e})")

# --- numerical round-trip ---------------------------------------------------
def get(name):
    return np.asarray(T[name].data, dtype=np.float32)

payload = torch.load(PT, map_location="cpu", weights_only=True)
sd = payload["mtp_heads"]

ck = torch.load(PTH, map_location="cpu", weights_only=True, mmap=True)
emb = ck["emb.weight"].float()

C = payload["n_embd"]
eps = 1e-5
tok_id = 1234


def layernorm(x, w, b):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# --- test inputs ------------------------------------------------------------
h_t = torch.randn(1, 5, C) * 0.5
tok_t = emb[tok_id].reshape(1, 1, C).expand(1, 5, C).contiguous()
h = h_t[0].numpy()                       # (5, C)
tok = tok_t[0].numpy()                   # (5, C)
Tn = h.shape[0]

if _pt_arch == "mlp":
    # --- reference: the actual trained torch module ---
    import sys
    sys.argv = ["x"]
    from mtp import MTPModule

    m = MTPModule(C)
    m.load_state_dict({k[2:]: v for k, v in sd.items() if k.startswith("0.")})
    m = m.float().eval()
    with torch.no_grad():
        z_ref = m(h_t, tok_t)[0].numpy()

    # --- gguf-side: exactly the graph the C++ must build ---
    h_norm = layernorm(h, get("blk.0.nextn.hnorm.weight"), get("blk.0.nextn.hnorm.bias"))
    e_norm = layernorm(tok, get("blk.0.nextn.enorm.weight"), get("blk.0.nextn.enorm.bias"))
    concat = np.concatenate([h_norm, e_norm], axis=-1)      # [hidden, tok] -- hidden FIRST

    # ggml stores Linear(in,out); numpy matmul wants (out,in)
    z0 = concat @ get("blk.0.nextn.eh_proj.weight").T
    mn = layernorm(z0, get("blk.0.nextn.mlp_norm.weight"), get("blk.0.nextn.mlp_norm.bias"))
    up = mn @ get("blk.0.nextn.mlp_up.weight").T
    z = z0 + silu(up) @ get("blk.0.nextn.mlp_down.weight").T
else:
    # --- rwkv7 head ---------------------------------------------------------
    # reference: pure-pytorch port of the trained math (mtp.RWKV7MTPModule /
    # spec_decode.RWKV7HeadRunner._run, wkv7 recurrence unrolled), reading the
    # CHECKPOINT tensors directly with their PyTorch shapes.
    import sys
    sys.argv = ["x"]

    sd0 = {k[2:]: v.float() for k, v in sd.items() if k.startswith("0.")}
    Ck = sd0

    def tln(x, p):
        return torch.nn.functional.layer_norm(
            x, (x.shape[-1],), Ck[p + ".weight"], Ck[p + ".bias"], eps)

    def twkv7(r3, w3, k3, v3, a3, b3, state):
        # exact training/CPU-kernel recurrence, fp32.  All (T, H, N); state (H,N,N).
        Tt, H, N = r3.shape
        st = state.clone()
        ys = []
        for t in range(Tt):
            sa = torch.einsum("hj,hij->hi", a3[t], st)
            st = (st * w3[t].unsqueeze(1)
                  + torch.einsum("hi,hj->hij", v3[t], k3[t])
                  + sa.unsqueeze(-1) * b3[t].unsqueeze(-1))
            ys.append(torch.einsum("hij,hj->hi", st, r3[t]))
        return torch.stack(ys), st

    def head_torch(h_rows, tok_rows):
        F = torch.nn.functional
        # fusion (hidden FIRST)
        x = torch.cat([tln(h_rows, "ln_h"), tln(tok_rows, "ln_tok")], -1) \
            @ Ck["proj.weight"].t()
        # Block(args, 0).forward(x, None) -- layer-0 semantics
        base = tln(x, "block.ln0")
        xn = tln(base, "block.ln1")
        xp = torch.cat([torch.zeros(1, C), xn[:-1]], 0)   # fresh token-shift state
        d = xp - xn
        vec = lambda p: Ck[p].reshape(-1)
        xr = xn + d * vec("block.att.x_r")
        xw = xn + d * vec("block.att.x_w")
        xk = xn + d * vec("block.att.x_k")
        xv = xn + d * vec("block.att.x_v")
        xa = xn + d * vec("block.att.x_a")
        xg = xn + d * vec("block.att.x_g")

        r = xr @ Ck["block.att.receptance.weight"].t()
        # decay: w_pre = -softplus(-(w0 + w_m)) - 0.5, decay = exp(-exp(w_pre))
        wm = torch.tanh(xw @ Ck["block.att.w1"]) @ Ck["block.att.w2"]
        w_pre = -F.softplus(-(vec("block.att.w0") + wm)) - 0.5
        w_dec = torch.exp(-torch.exp(w_pre))
        k = xk @ Ck["block.att.key.weight"].t()
        v = xv @ Ck["block.att.value.weight"].t()
        a = torch.sigmoid(vec("block.att.a0") + (xa @ Ck["block.att.a1"]) @ Ck["block.att.a2"])
        g = torch.sigmoid(xg @ Ck["block.att.g1"]) @ Ck["block.att.g2"]

        H = Ck["block.att.r_k"].shape[0]
        N = C // H
        kk = F.normalize((k * vec("block.att.k_k")).view(Tn, H, N), dim=-1, p=2.0).view(Tn, C)
        k2 = k * (1 + (a - 1) * vec("block.att.k_a"))

        y, _ = twkv7(r.view(Tn, H, N), w_dec.view(Tn, H, N), k2.view(Tn, H, N),
                     v.view(Tn, H, N), (-kk).view(Tn, H, N), (kk * a).view(Tn, H, N),
                     torch.zeros(H, N, N))
        # group norm over heads (eps=64e-5), rk residual, gate, output
        y = F.group_norm(y.reshape(Tn, C), H, Ck["block.att.ln_x.weight"], Ck["block.att.ln_x.bias"], 64e-5)
        y = y.view(Tn, H, N)
        rk = (r.view(Tn, H, N) * k2.view(Tn, H, N) * Ck["block.att.r_k"]).sum(-1, keepdim=True)
        y = (y + rk * v.view(Tn, H, N)).view(Tn, C)
        y = y * g
        att_out = y @ Ck["block.att.output.weight"].t()

        ffn_inp = base + att_out
        xn2 = tln(ffn_inp, "block.ln2")
        xp2 = torch.cat([torch.zeros(1, C), xn2[:-1]], 0)
        k2f = xn2 + (xp2 - xn2) * vec("block.ffn.x_k")
        k2f = torch.relu(k2f @ Ck["block.ffn.key.weight"].t()) ** 2
        z_out = (k2f @ Ck["block.ffn.value.weight"].t()) + ffn_inp
        return z_out

    with torch.no_grad():
        z_ref = head_torch(h_t[0], tok_t[0]).numpy()

    # --- gguf-side: exactly the graph the C++ must build (recurrent state) ---
    g = lambda n: get("blk.0.nextn." + n)
    # head_size comes from the gguf metadata (rwkv7.wkv.head_size)
    N = int(r.fields["rwkv7.wkv.head_size"].parts[-1][0])
    H = C // N

    h_norm = layernorm(h, g("hnorm.weight"), g("hnorm.bias"))
    e_norm = layernorm(tok, g("enorm.weight"), g("enorm.bias"))
    concat = np.concatenate([h_norm, e_norm], axis=-1)      # [hidden, tok] -- hidden FIRST
    x = concat @ g("eh_proj.weight").T

    base = layernorm(x, g("block_ln0.weight"), g("block_ln0.bias"))
    xn = layernorm(base, g("attn_norm.weight"), g("attn_norm.bias"))
    xp = np.concatenate([np.zeros((1, C)), xn[:-1]], 0)     # fresh token-shift state
    d = xp - xn
    lerp = g("time_mix_lerp_fused.weight").reshape(6, C)    # r,w,k,v,a,g
    xr = xn + d * lerp[0]
    xw = xn + d * lerp[1]
    xk = xn + d * lerp[2]
    xv = xn + d * lerp[3]
    xa = xn + d * lerp[4]
    xg = xn + d * lerp[5]

    r_mat = xr @ g("time_mix_receptance.weight").T
    # the C++ graph uses the algebraically identical exp(-0.606531*sigmoid(w)) form
    wm = np.tanh(xw @ g("time_mix_w1.weight").T) @ g("time_mix_w2.weight").T
    w_dec = np.exp(sigmoid(g("time_mix_w0.weight") + wm) * -0.606531)
    k = xk @ g("time_mix_key.weight").T
    v = xv @ g("time_mix_value.weight").T
    a = sigmoid(g("time_mix_a0.weight") + (xa @ g("time_mix_a1.weight").T) @ g("time_mix_a2.weight").T)
    gg_ = sigmoid(xg @ g("time_mix_g1.weight").T) @ g("time_mix_g2.weight").T

    def l2norm_rows(m, eps_n=1e-12):
        n = np.linalg.norm(m, axis=-1, keepdims=True)
        return m / np.maximum(n, eps_n)

    kk = l2norm_rows((k * g("time_mix_k_k.weight")).reshape(Tn, H, N)).reshape(Tn, C)
    k2 = k * (1 + (a - 1) * g("time_mix_k_a.weight"))

    def wkv7_np(r3, w3, k3, v3, a3, b3, state):
        # exact port of ggml_compute_forward_rwkv_wkv7_f32
        st = state.copy()
        ys = np.empty((Tn, C), dtype=np.float64)
        for t in range(Tn):
            sa = np.einsum("hj,hij->hi", a3[t], st)
            st = (st * w3[t][:, None, :]
                  + np.einsum("hi,hj->hij", v3[t], k3[t])
                  + sa[:, :, None] * b3[t][:, :, None])
            ys[t] = np.einsum("hij,hj->hi", st, r3[t]).reshape(C)
        return ys, st

    y, _ = wkv7_np(r_mat.reshape(Tn, H, N), w_dec.reshape(Tn, H, N), k2.reshape(Tn, H, N),
                   v.reshape(Tn, H, N), (-kk).reshape(Tn, H, N),
                   (kk * a).reshape(Tn, H, N), np.zeros((H, N, N)))
    # group norm per (t, h) over head_size, then per-channel scale/shift
    y3 = y.reshape(Tn, H, N)
    mu = y3.mean(-1, keepdims=True)
    var = y3.var(-1, keepdims=True)          # biased, like ggml_norm
    y3 = (y3 - mu) / np.sqrt(var + 64e-5)
    y = y3.reshape(Tn, C) * g("time_mix_ln.weight") + g("time_mix_ln.bias")
    rk = (r_mat.reshape(Tn, H, N) * k2.reshape(Tn, H, N)
          * g("time_mix_r_k.weight").reshape(H, N)).sum(-1, keepdims=True)
    y = (y.reshape(Tn, H, N) + rk * v.reshape(Tn, H, N)).reshape(Tn, C)
    y = y * gg_
    att_out = y @ g("time_mix_output.weight").T

    ffn_inp = base + att_out
    xn2 = layernorm(ffn_inp, g("attn_norm_2.weight"), g("attn_norm_2.bias"))
    xp2 = np.concatenate([np.zeros((1, C)), xn2[:-1]], 0)
    k2f = xn2 + (xp2 - xn2) * g("channel_mix_lerp_k.weight")
    k2f = np.maximum(k2f @ g("channel_mix_key.weight").T, 0.0) ** 2
    z = (k2f @ g("channel_mix_value.weight").T) + ffn_inp

err = np.abs(z - z_ref)
den = np.abs(z_ref)
print("\n=== MTP head numeric round-trip (gguf f16 vs torch fp32 from checkpoint) ===")
print(f"  arch               {_pt_arch}")
print(f"  max |abs err|      {err.max():.3e}")
print(f"  mean|abs err|      {err.mean():.3e}")
print(f"  max rel err        {(err / (den + 1e-6)).max():.3e}")
print(f"  z std (reference)  {z_ref.std():.4f}   gguf {z.std():.4f}")

# --- verify the lm-head duplicate matches the frozen head ------------------
# NB: GGUFReader exposes .shape in ggml ne-order but .data reshaped to
# reversed(ne) == PyTorch order, so .data is directly comparable to HF tensors.
out = get("output.weight")
head = ck["head.weight"].float().numpy()
d = np.abs(out - head)
print("\n=== output.weight vs frozen head.weight ===")
print(f"  ggml ne-order (t.shape) {tuple(int(x) for x in T['output.weight'].shape)}")
print(f"  .data (PyTorch order)   {out.shape}, head {head.shape}")
print(f"  max |abs err| {d.max():.3e}  (f16 rounding)")

te = get("token_embd.weight")
e = ck["emb.weight"].float().numpy()
print(f"  token_embd max |abs err| vs emb.weight: {np.abs(te - e).max():.3e}")

# --- full contract: the draft's logits use the SAME frozen head ------------
# Isolate the head: same z on both sides, so this measures only the duplicated
# output.weight. (Feeding the gguf z through would just amplify the ~2e-05
# hidden-state rounding by the head's dynamic range, which tests nothing.)
z_t = torch.from_numpy(z_ref)
with torch.no_grad():
    logits_ref = torch.nn.functional.linear(z_t, ck["head.weight"].float()).numpy()
logits_gguf = z_ref @ get("output.weight").T
assert logits_ref.shape == logits_gguf.shape, (logits_ref.shape, logits_gguf.shape)
ld = np.abs(logits_gguf - logits_ref)
print("\n=== draft logits = output @ z (same z, head duplication only) ===")
print(f"  logits: std {logits_ref.std():.2f} min {logits_ref.min():.2f} max {logits_ref.max():.2f}")
print(f"  max |abs err| {ld.max():.3e}   mean {ld.mean():.3e}")
print(f"  argmax match: {(logits_gguf.argmax(-1) == logits_ref.argmax(-1)).all()}  "
      f"({logits_gguf.shape[0]} positions)")

# --- tokenizer -------------------------------------------------------------
def s(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", "replace")
    if isinstance(x, np.ndarray):
        return "[" + ", ".join(s(v) for v in x.tolist()) + "]"
    return str(x)


toks = r.fields["tokenizer.ggml.tokens"].data
print(f"\n=== tokenizer ===")
print(f"  model={s(r.fields['tokenizer.ggml.model'].data)!r}  n_tokens={len(toks)}")
print(f"  first={s(toks[0])!r}   last={s(toks[-1])!r}")
print(f"  bos={s(r.fields['tokenizer.ggml.bos_token_id'].data)} "
      f"eos={s(r.fields['tokenizer.ggml.eos_token_id'].data)} "
      f"eot={s(r.fields['tokenizer.ggml.eot_token_id'].data)}")

ok = (err.max() < 5e-3 and d.max() < 5e-3 and ld.max() < 5e-3
      and len(toks) == 65536 and probe in T and bc - nn == 0)
print("\nOK" if ok else "\nFAIL")
