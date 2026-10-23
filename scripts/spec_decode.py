"""
spec_decode.py -- speculative decoding with the trained MTP heads.

Draft/verify loop (greedy by default, which is exact):

  1. From the backbone hidden h at position p, the shared head proposes t1.
  2. Chain the MTP heads: z_k = MTP_k(h_{k-1}, Emb(t_k)) predicts t_{k+1}.
     With K heads we draft t1..t_{K+1}.
  3. Run the backbone once over the K+1 drafts (batched, RNN mode) to get the
     target's own prediction at each drafted position.
  4. Accept the longest matching prefix; take the target's token at the first
     mismatch as a bonus. Commit the accepted tokens + bonus to the RNN state.

Two head architectures are supported (--mtp-arch, auto-detected from the
checkpoint by default):

* ``mlp`` -- the position-local head.  Stateless across draft steps: each chain
  step only needs (previous output vector, last token embedding).
* ``rwkv7`` -- one full RWKV-7 block as the head.  It mixes across stream
  positions, so it carries a true recurrent state: one WKV state per draft
  depth, advanced over stream position j as soon as the token at j+k+1 is
  committed (RWKV7HeadRunner below).  This reproduces the training-time
  parallel form exactly, including rollback when verification rejects.

Reports per-depth acceptance and end-to-end tokens/s vs. plain single-token
decoding.

Usage:
    python scripts/spec_decode.py --mtp mtp_out/mtp_heads.pt --num-mtp 3 \
        --max-new 128 --device cuda:0
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
import time

import torch
import torch.nn as nn

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from generate_data import BatchedRWKV7, RWKV_TOKENIZER, sample_batch, _load_batched_kernel  # noqa: E402
from mtp import MTPModule  # noqa: E402


# PROMPTS = [
    # "User: Give me three uses of a paperclip.\n\nAssistant:",
    # "User: Explain recursion to a five year old.\n\nAssistant:",
    # "User: Write a haiku about the ocean.\n\nAssistant:",
    # "User: What is the capital of France?\n\nAssistant:",
# ]

PROMPTS = [
    "User: Provide a Python code snippet for a function called two_sum that takes a list of integers nums and an integer target, and returns the indices of the two numbers that add up to target. Assume exactly one solution exists and you may not use the same element twice. Use a dictionary to achieve O(n) time complexity. Include a docstring and a short example of how to call the function.\n\nAssistant:"
]

# ---------------------------------------------------------------------------
# RWKV-7 head: recurrent (RNN-mode) runner
# ---------------------------------------------------------------------------

# nn.Linear weights are stored (out, in); the fp16 inference math uses x @ W,
# so these are transposed on load (same convention as BatchedRWKV7).
_RWKV7_HEAD_TRANSPOSE = (
    "fc_fuse.weight",
    "proj.weight",
    "block.att.receptance.weight",
    "block.att.key.weight",
    "block.att.value.weight",
    "block.att.output.weight",
    "block.ffn.key.weight",
    "block.ffn.value.weight",
)


def prepare_rwkv7_head_weights(sd: dict, device: str) -> dict:
    """Module-level MTP-head state dict -> fp16 inference weights."""
    w = {}
    for k, v in sd.items():
        v = v.detach().to(device=device, dtype=torch.half)
        if k in _RWKV7_HEAD_TRANSPOSE:
            v = v.t().contiguous()
        elif v.dim() == 3 and v.shape[0] == 1 and v.shape[1] == 1:
            # x_r..x_g, w0, a0, k_k, k_a are stored (1,1,C); BatchedRWKV7
            # squeezes them so the token-shift mixes broadcast (T,C)*(C,).
            v = v.squeeze()
        w[k] = v
    return w


def split_head_sd(sd: dict, n_modules: int) -> list:
    """Split an mtp_heads state dict into per-module dicts ("0.xxx" -> xxx)."""
    outs = [{} for _ in range(n_modules)]
    for k, v in sd.items():
        head, _, rest = k.partition(".")
        if head.isdigit() and rest:
            idx = int(head)
        else:
            idx, rest = 0, k
        if not (0 <= idx < n_modules):
            raise KeyError(f"head module index {idx} out of range ({n_modules} modules)")
        outs[idx][rest] = v
    return outs


class RWKV7HeadRunner:
    """Recurrent state machine for the ``--mtp-arch rwkv7`` head (fp16).

    Training (mtp.RWKV7MTPModule, parallel form) computes depth k by running
    the shared block over the per-depth input sequence

        x_k[j] = proj([ln_h(z_{k-1,j}) ; ln_tok(emb(s[j+k+1]))])

    where z_{-1,j} is the backbone hidden h_j and s[.] are stream tokens.  The
    block's WKV state at position j summarizes x_k[j'] for j' <= j, so at
    inference the state for depth k can be advanced over stream position j as
    soon as token s[j+k+1] is *committed* -- one state per depth, each token
    ingested exactly once per depth.  z_k[j] is kept in a small rolling buffer
    (zbufs) because depth k+1 needs it at the same position one commit later.

    The draft chain additionally ingests *tentative* inputs (embeddings of
    not-yet-verified drafts) into working copies of the states; snapshot()
    before the chain and restore() after verification roll those back, and
    advance() then re-ingests everything that became final with committed
    tokens -- reproducing the training computation exactly, accepted or not.
    """

    def __init__(self, sds, n_embd, head_size, emb, device="cuda:0", n_depths=None):
        self.device = device
        self.C = n_embd
        self.N = head_size
        self.H = n_embd // head_size
        self.K = int(n_depths or len(sds))   # number of draft depths
        self.emb = emb                 # (V, C) fp16, unfused embedding
        self.wsets = [prepare_rwkv7_head_weights(sd, device) for sd in sds]
        if len(self.wsets) == 1 and self.K > 1:
            # shared head: same block per depth -- but per-depth dicts, so the
            # ln_h overrides below don't alias one shared dict.
            self.wsets = [dict(self.wsets[0]) for _ in range(self.K)]
        # EAGLE-3 per-depth input normalization: depth 0 normalizes the fused
        # anchor with ln_h; depths k >= 1 use ln_h_extra[k-1] (training passes
        # depth=k to RWKV7MTPModule.forward).  Fold the per-depth affine into
        # each depth's weight set so _fuse stays untouched.
        for k in range(self.K):
            wk = self.wsets[min(k, len(self.wsets) - 1)]
            if k > 0 and f"ln_h_extra.{k - 1}.weight" in wk:
                wk["ln_h.weight"] = wk[f"ln_h_extra.{k - 1}.weight"]
                wk["ln_h.bias"] = wk[f"ln_h_extra.{k - 1}.bias"]
        _load_batched_kernel(head_size)
        C, H, N = self.C, self.H, self.N
        # One WKV state per depth: [x_prev (C,), kv (H,N,N) fp32, x2_prev (C,)].
        self.st = [
            [torch.zeros(1, C, dtype=torch.half, device=device),
             torch.zeros(1, H, N, N, dtype=torch.float32, device=device),
             torch.zeros(1, C, dtype=torch.half, device=device)]
            for _ in range(self.K)
        ]
        self.zbufs = [{} for _ in range(self.K)]   # depth -> {pos j: z (1,C)}
        self.wm = [-1] * self.K                    # per-depth ingest watermark

    # -- core math ---------------------------------------------------------

    def _fuse(self, w, h, tok_emb):
        """(T,C) + (T,C) -> fused residual-stream input (T,C)."""
        hn = torch.nn.functional.layer_norm(h, (self.C,), w["ln_h.weight"], w["ln_h.bias"])
        tn = torch.nn.functional.layer_norm(tok_emb, (self.C,), w["ln_tok.weight"], w["ln_tok.bias"])
        return torch.cat([hn, tn], dim=-1) @ w["proj.weight"]

    def _run(self, w, X, st):
        """One block application over T fused inputs; mutates st in place.

        Mirrors generate_data.BatchedRWKV7's per-layer math (fp16, RNN mode)
        for a layer-0 block: v_first = own value (no value residual).
        """
        F = torch.nn.functional
        C, H, N = self.C, self.H, self.N
        T = X.shape[0]
        attw = {k[10:]: v for k, v in w.items() if k.startswith("block.att.")}
        ffnw = {k[10:]: v for k, v in w.items() if k.startswith("block.ffn.")}

        base = F.layer_norm(X, (C,), w["block.ln0.weight"], w["block.ln0.bias"])
        xn = F.layer_norm(base, (C,), w["block.ln1.weight"], w["block.ln1.bias"])
        xp = torch.cat([st[0], xn[:-1]], dim=0) if T > 1 else st[0]
        d = xp - xn
        xr = xn + d * attw["x_r"]
        xw = xn + d * attw["x_w"]
        xk = xn + d * attw["x_k"]
        xv = xn + d * attw["x_v"]
        xa = xn + d * attw["x_a"]
        xg = xn + d * attw["x_g"]

        r = xr @ attw["receptance.weight"]
        w_m = torch.tanh(xw @ attw["w1"]) @ attw["w2"]
        k = xk @ attw["key.weight"]
        v = xv @ attw["value.weight"]
        a = torch.sigmoid(attw["a0"] + (xa @ attw["a1"]) @ attw["a2"])
        g = torch.sigmoid(xg @ attw["g1"]) @ attw["g2"]

        kk = F.normalize((k * attw["k_k"]).view(T, H, N), dim=-1, p=2.0).view(T, C)
        k = k * (1 + (a - 1) * attw["k_a"])
        w_pre = -F.softplus(-(attw["w0"] + w_m)) - 0.5
        y = torch.empty((1, T, C), device=self.device, dtype=torch.half)
        torch.ops.wkv7sb.forward(
            1, T, C, H, st[1],
            r.contiguous(), w_pre.contiguous(), k.contiguous(), v.contiguous(),
            (-kk).contiguous(), (kk * a).contiguous(), y)

        y = F.group_norm(y.view(T, C), num_groups=H,
                         weight=attw["ln_x.weight"], bias=attw["ln_x.bias"],
                         eps=64e-5).view(T, H, N)
        y = y + (r.view(T, H, N) * k.view(T, H, N) * attw["r_k"].view(1, H, N)
                 ).sum(dim=-1, keepdim=True) * v.view(T, H, N)
        y = (y.view(T, C) * g) @ attw["output.weight"]
        x = base + y

        xn2 = F.layer_norm(x, (C,), w["block.ln2.weight"], w["block.ln2.bias"])
        xp2 = torch.cat([st[2], xn2[:-1]], dim=0) if T > 1 else st[2]
        d2 = xp2 - xn2
        k2 = xn2 + d2 * ffnw["x_k"]
        k2 = torch.relu(k2 @ ffnw["key.weight"]) ** 2
        x = x + k2 @ ffnw["value.weight"]
        st[0] = xn[-1:, :].contiguous()
        st[2] = xn2[-1:, :].contiguous()
        if "ln_out.weight" in w:
            # EAGLE-3-era heads end with their own final LayerNorm (training
            # returns ln_out(z)); older checkpoints don't have it and stay
            # bit-identical to the pre-EAGLE-3 math.
            x = F.layer_norm(x, (C,), w["ln_out.weight"], w["ln_out.bias"])
        return x

    def fuse_taps(self, h_early: torch.Tensor, h_mid: torch.Tensor,
                  h_late: torch.Tensor) -> torch.Tensor:
        """Fuse (T, C) tap tensors into (T, C) depth-0 input h0.

        Mirrors RWKV7MTPModule.fuse_backbone: separate LayerNorms on the early
        and middle taps, raw post-ln_out final tap, then the [0, 0, I]-initialized
        fc_fuse projection.  Checkpoints without fusion weights return h_late
        unchanged (pre-EAGLE-3 behavior).
        """
        w = self.wsets[0]
        if "fc_fuse.weight" not in w:
            return h_late
        F = torch.nn.functional
        he = F.layer_norm(h_early, (self.C,), w["ln_early.weight"], w["ln_early.bias"])
        hm = F.layer_norm(h_mid, (self.C,), w["ln_mid.weight"], w["ln_mid.bias"])
        return torch.cat([he, hm, h_late], dim=-1) @ w["fc_fuse.weight"]

    def _ingest(self, k, prev_rows, toks):
        """Ingest x_k[j] for consecutive j; returns the z rows (T, C).

        prev_rows: list of (1,C) tensors (backbone hidden h_j for k=0,
        z_{k-1,j} otherwise); toks: the committed-or-drafted token ids at
        j+k+1 for each j.
        """
        w = self.wsets[k if k < len(self.wsets) else 0]
        H_rows = torch.cat(prev_rows, dim=0)
        tok_rows = self.emb[torch.tensor(toks, device=self.device)]
        X = self._fuse(w, H_rows, tok_rows)
        return self._run(w, X, self.st[k])

    # -- lifecycle ---------------------------------------------------------

    def prefill(self, H, tokens):
        """Advance all depth states over a fully-committed prefix.

        H: (P, C) fp16 depth-0 state rows for stream positions 0..P-1 -- the
        fused EAGLE-3 anchors when the checkpoint has fusion (fuse_taps over
        the whole prefix), else the raw backbone hiddens; tokens:
        the P committed token ids.  Afterwards depth k has ingested
        x_k[j] for j <= P-k-2 (its input needs token j+k+1, which must exist).
        """
        P = len(tokens)
        p = P - 1
        for k in range(self.K):
            lo, hi = 0, p - k - 1
            if hi < lo:
                continue
            prevs = [H[j:j + 1] if k == 0 else self.zbufs[k - 1][j]
                     for j in range(lo, hi + 1)]
            toks = [tokens[j + k + 1] for j in range(lo, hi + 1)]
            Z = self._ingest(k, prevs, toks)
            for off, j in enumerate(range(lo, hi + 1)):
                self.zbufs[k][j] = Z[off:off + 1]
            self.wm[k] = hi

    def snapshot(self):
        return {
            "st": [[t.clone() for t in st] for st in self.st],
            "zbufs": [{j: z.clone() for j, z in buf.items()} for buf in self.zbufs],
            "wm": list(self.wm),
        }

    def restore(self, snap):
        for k in range(self.K):
            for i in range(3):
                self.st[k][i].copy_(snap["st"][k][i])
            self.zbufs[k] = {j: z.clone() for j, z in snap["zbufs"][k].items()}
            self.wm[k] = snap["wm"][k]

    def draft_chain(self, h, p, pick, h0=None):
        """Produce the K+1 drafts at stream end p; mutates states tentatively.

        h: (1,C) backbone hidden H[p] -- picks the backbone's own next token
        (drafts[0]); h0: (1,C) EAGLE-3 fused depth-0 anchor at p (fuse_taps of
        the early/middle/final taps), used as depth 0's state input.  With
        h0=None (or a pre-fusion checkpoint) depth 0 falls back to h.
        pick: callable (1,C) -> token tensor.
        Besides the chain inputs at j=p (embeddings of the tentative drafts),
        each depth k first catch-ups on x_k[p-k], whose token (c_{p+1} =
        drafts[0]) is always committed and whose z_{k-1}[p-k] is the
        watermark entry -- so these ingestions are exact.  Callers snapshot()
        before and restore()+advance() after verification.
        """
        drafts = [pick(h)]
        zloc = [{} for _ in range(self.K)]   # chain-local z by position
        z_end = None
        for k in range(self.K):
            for j in range(p - k, p + 1):
                if k == 0:
                    prev = h0 if h0 is not None else h
                elif j == p - k:
                    prev = self.zbufs[k - 1][j]          # watermark entry
                else:
                    prev = zloc[k - 1][j]                # from this chain
                # drafts[i] proposes stream token c_{p+1+i}; x_k[j] wants c_{j+k+1}
                tok = int(drafts[j + k - p].item())
                zloc[k][j] = self._ingest(k, [prev], [tok])
            z_end = zloc[k][p]
            drafts.append(pick(z_end))
        return drafts

    def advance(self, h_end, H_new, p, tokens):
        """Roll back nothing; caller restored first.  Re-ingest with committed data.

        h_end: (1,C) = the depth-0 state row at p (fused h0[p] with EAGLE-3
        fusion, else backbone H[p]) -- i.e. H[p] before the commit; H_new: (S,C)
        = the same kind of rows for H[p+1..p'] (the commit's fused hiddens);
        tokens: the committed stream, len >= p'+1.  Depth k ingests x_k[j] for
        j in [p-k, p'-k-1] -- exactly the inputs that became final, including
        the formerly-tentative ones (whose committed tokens equal the drafts
        when they were accepted).
        """
        q = p + H_new.shape[0]               # new stream end p'
        for k in range(self.K):
            lo, hi = p - k, q - k - 1
            if hi < lo:
                continue
            prevs = []
            for j in range(lo, hi + 1):
                if k == 0:
                    prevs.append(h_end if j == p else H_new[j - p - 1:j - p])
                else:
                    prevs.append(self.zbufs[k - 1][j])   # snapshot or depth k-1
            toks = [tokens[j + k + 1] for j in range(lo, hi + 1)]
            Z = self._ingest(k, prevs, toks)
            for off, j in enumerate(range(lo, hi + 1)):
                self.zbufs[k][j] = Z[off:off + 1]
            self.wm[k] = hi
            # bound the buffer: the deepest lookback is depth K-1's catch-up
            oldest = q - self.K - 8
            if oldest > 0:
                for j in [j for j in self.zbufs[k] if j < oldest]:
                    del self.zbufs[k][j]


# ---------------------------------------------------------------------------
# Speculative decoder
# ---------------------------------------------------------------------------


class SpecDecoder:
    def __init__(self, pth, config, mtp_path, device="cuda:0", arch=None):
        with open(config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.device = device
        self.n_embd = cfg["hidden_size"]
        self.head_size = cfg.get("head_dim", 64)
        self.model = BatchedRWKV7(pth, cfg["num_hidden_layers"],
                                  cfg["hidden_size"], cfg["head_dim"], device,
                                  keep_unfused=True)
        self.head = self.model.z["head.weight"]     # (C, V), fp16
        self.emb = self.model.emb_unfused           # (V, C), fp16

        payload = torch.load(mtp_path, map_location="cpu", weights_only=True)
        sd = payload["mtp_heads"] if "mtp_heads" in payload else payload
        self.num_mtp = payload.get("num_mtp")
        if self.num_mtp is None:
            self.num_mtp = len({k.split(".")[0] for k in sd})
        self.num_mtp = int(self.num_mtp)
        # FastMTP-style shared head: one module, applied num_mtp times.
        self.shared_head = bool(payload.get("shared_head", False))
        ckpt_arch = str(payload.get("arch")
                        or (payload.get("meta") or {}).get("arch") or "mlp")
        self.eagle3_fusion = bool(
            payload.get("eagle3_fusion",
                        (payload.get("meta") or {}).get("eagle3_fusion", False)))
        self.arch = arch or ckpt_arch
        if self.arch != ckpt_arch:
            raise ValueError(
                f"--mtp-arch {self.arch} but checkpoint {mtp_path} was trained "
                f"with --mtp-arch {ckpt_arch}")
        n_modules = 1 if self.shared_head else self.num_mtp
        if self.arch == "rwkv7":
            sds = split_head_sd(sd, n_modules)
            self.runner = RWKV7HeadRunner(sds, self.n_embd, self.head_size,
                                          self.emb, self.device,
                                          n_depths=self.num_mtp)
            self.mtps = None
        else:
            self.mtps = nn.ModuleList([MTPModule(self.n_embd) for _ in range(n_modules)])
            self.mtps.load_state_dict(sd)
            self.mtps.to(device=device, dtype=torch.half).eval()
            self.runner = None

    def _pick(self, logits, greedy, temp, top_p, top_k, gen):
        if greedy:
            return logits.argmax(dim=-1)
        return sample_batch(logits, temp, top_p, top_k, gen)

    @torch.no_grad()
    def _draft(self, h, greedy, temp, top_p, top_k, gen):
        drafts = []
        t = self._pick(h @ self.head, greedy, temp, top_p, top_k, gen)
        drafts.append(t)
        h_prev, ztok = h, self.emb[t]
        for k in range(self.num_mtp):
            mtp = self.mtps[0] if self.shared_head else self.mtps[k]
            z = mtp(h_prev, ztok)
            t = self._pick(z @ self.head, greedy, temp, top_p, top_k, gen)
            drafts.append(t)
            h_prev, ztok = z, self.emb[t]
        return drafts  # length K+1, each (1,)

    @torch.no_grad()
    def generate(self, prompt_ids, max_new, greedy=True, temp=0.8, top_p=0.9,
                 top_k=0, seed=0):
        gen = torch.Generator(device=self.device).manual_seed(seed)
        idx = torch.tensor(prompt_ids, device=self.device).unsqueeze(0)
        # EAGLE-3 fusion: the backbone forward must also yield the early/middle
        # taps so the fused depth-0 anchors (h0) can be built for every stream
        # position.  `h` stays the raw post-ln_out hidden (picks drafts[0] and
        # feeds the main head); `h0` is the fused depth-0 state input.
        want_taps = self.arch == "rwkv7" and self.eagle3_fusion \
            and self.runner is not None

        def _fused_seq(hidden_tc, taps):
            """(T,C) hidden + taps -> (T,C) depth-0 rows (h0 == hidden w/o fusion)."""
            if not want_taps:
                return hidden_tc
            return self.runner.fuse_taps(taps[0], taps[1], hidden_tc)

        if want_taps:
            hidden, _, state, taps = self.model.forward(
                idx, None, return_hidden=True, return_taps=True)
        else:
            hidden, _, state = self.model.forward(idx, None, return_hidden=True)
        h = hidden[:, -1, :]
        rwkv7 = self.arch == "rwkv7"
        if rwkv7:
            H0 = _fused_seq(hidden[0], None if not want_taps
                            else (taps[0][0], taps[1][0]))
            self.runner.prefill(H0.half() if H0.dtype != torch.half else H0,
                                list(prompt_ids))
            h0_p = H0[-1:]                      # fused anchor at stream end p

        out = list(prompt_ids)
        start = len(out)
        acc = [0] * self.num_mtp
        att = [0] * self.num_mtp
        steps = 0

        while len(out) - start < max_new:
            steps += 1
            p = len(out) - 1
            if rwkv7:
                snap = self.runner.snapshot()
                pick = lambda zz: self._pick(zz @ self.head, greedy, temp, top_p, top_k, gen)
                drafts = self.runner.draft_chain(h, p, pick,
                                                 h0=h0_p if want_taps else None)
            else:
                drafts = self._draft(h, greedy, temp, top_p, top_k, gen)
            draft_t = torch.cat(drafts, dim=0)              # (L,)
            L = draft_t.numel()

            if want_taps:
                hidden_v, logits_v, state_v, taps_v = self.model.forward(
                    draft_t.view(1, L), state, return_hidden=True, return_taps=True)
            else:
                hidden_v, logits_v, state_v = self.model.forward(
                    draft_t.view(1, L), state, return_hidden=True)
            preds = self._pick(logits_v[0], greedy, temp, top_p, top_k, gen)  # (L,)

            m = 0
            while m < L - 1 and preds[m].item() == draft_t[m + 1].item():
                m += 1
            for j in range(L - 1):
                if j <= m:
                    att[j] += 1
                    if j < m:
                        acc[j] += 1

            emitted = draft_t[: m + 1].tolist() + [int(preds[m].item())]
            if m == L - 1:
                # state_v already consumed every draft; only the bonus remains
                bonus = torch.tensor([[int(preds[m].item())]], device=self.device)
                if want_taps:
                    hidden2, _, state, taps2 = self.model.forward(
                        bonus, state_v, return_hidden=True, return_taps=True)
                else:
                    hidden2, _, state = self.model.forward(
                        bonus, state_v, return_hidden=True)
            else:
                commit = torch.tensor([emitted], device=self.device)
                if want_taps:
                    hidden2, _, state, taps2 = self.model.forward(
                        commit, state, return_hidden=True, return_taps=True)
                else:
                    hidden2, _, state = self.model.forward(
                        commit, state, return_hidden=True)

            for tk in emitted:
                out.append(tk)

            if rwkv7:
                # Roll back the tentative chain ingestions, then re-ingest
                # everything that became final with the committed tokens.
                if m == L - 1:
                    H_new = torch.cat([hidden_v[0], hidden2[:, -1, :]], dim=0)
                    if want_taps:
                        # full acceptance: the verification rows carry the exact
                        # committed tokens (drafts == emitted), so taps_v aligns;
                        # only the bonus row comes from the bonus forward.
                        t_early = torch.cat([taps_v[0][0], taps2[0][0]], dim=0)
                        t_mid = torch.cat([taps_v[1][0], taps2[1][0]], dim=0)
                else:
                    H_new = hidden2[0]
                    if want_taps:
                        # partial acceptance: the commit forward re-ran the
                        # accepted prefix + target's bonus token, so ITS taps
                        # (not the verification taps, whose last rows carry the
                        # rejected draft token) are the committed ones.
                        t_early, t_mid = taps2[0][0], taps2[1][0]
                if want_taps:
                    H0_new = self.runner.fuse_taps(t_early, t_mid, H_new)
                else:
                    H0_new = H_new
                self.runner.restore(snap)
                self.runner.advance(h0_p if want_taps else h, H0_new.half()
                                    if H0_new.dtype != torch.half else H0_new,
                                    p, out)

            h = hidden2[:, -1, :]
            if want_taps:
                h0_p = self.runner.fuse_taps(taps2[0][0][-1:], taps2[1][0][-1:], h)

            for tk in emitted:
                if tk == 0:
                    return out, acc, att, steps
        return out, acc, att, steps

    @torch.no_grad()
    def generate_baseline(self, prompt_ids, max_new, greedy=True, temp=0.8,
                          top_p=0.9, top_k=0, seed=0):
        gen = torch.Generator(device=self.device).manual_seed(seed)
        idx = torch.tensor(prompt_ids, device=self.device).unsqueeze(0)
        logits, state = self.model.forward(idx, None)
        logits = logits[:, -1, :]
        out = list(prompt_ids)
        start = len(out)
        while len(out) - start < max_new:
            t = self._pick(logits, greedy, temp, top_p, top_k, gen)
            out.append(int(t.item()))
            if int(t.item()) == 0:
                break
            logits, state = self.model.forward(t.view(1, 1), state)
            logits = logits[:, -1, :]
        return out


def main():
    ap = argparse.ArgumentParser(description="Speculative decoding with MTP heads")
    ap.add_argument("--pth", default="rwkv7-g1k-2.9b-20260930-ctx25600.pth")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--mtp", default="mtp_out/mtp_heads.pt")
    ap.add_argument("--vocab", default="rwkv_vocab_v20230424.txt")
    ap.add_argument("--num-mtp", type=int, default=0, help="0 = read from checkpoint")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--greedy", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-prompts", type=int, default=4)
    ap.add_argument("--mtp-arch", default=None, choices=["mlp", "rwkv7"],
                    help="MTP head architecture; default: auto-detect from "
                         "the checkpoint (legacy checkpoints read as 'mlp')")
    args = ap.parse_args()

    tokenizer = RWKV_TOKENIZER(args.vocab)
    sd = SpecDecoder(args.pth, args.config, args.mtp, args.device,
                     arch=args.mtp_arch)
    print(f"[spec] loaded {sd.num_mtp} MTP depth(s) from {args.mtp} "
          f"(arch={sd.arch}{' , shared head' if sd.shared_head else ''})")

    prompts = [tokenizer.encode(p) for p in PROMPTS[: args.num_prompts]]

    # ---- baseline ---------------------------------------------------------
    t0 = time.time()
    base_tokens = 0
    for p in prompts:
        out = sd.generate_baseline(p, args.max_new, bool(args.greedy),
                                   args.temperature, args.top_p, seed=0)
        base_tokens += len(out) - len(p)
    base_dt = time.time() - t0
    print(f"[spec] baseline: {base_tokens} tokens in {base_dt:.1f}s = "
          f"{base_tokens / max(base_dt, 1e-6):.0f} tok/s")

    # ---- speculative ------------------------------------------------------
    t0 = time.time()
    spec_tokens = 0
    acc_tot = [0] * sd.num_mtp
    att_tot = [0] * sd.num_mtp
    steps_tot = 0
    for i, p in enumerate(prompts):
        out, acc, att, steps = sd.generate(p, args.max_new, bool(args.greedy),
                                           args.temperature, args.top_p, seed=0)
        spec_tokens += len(out) - len(p)
        steps_tot += steps
        for k in range(sd.num_mtp):
            acc_tot[k] += acc[k]
            att_tot[k] += att[k]
        if i == 0:
            print("[spec] sample:", tokenizer.decode(out[len(p):][:120]).replace("\n", " "))
    spec_dt = time.time() - t0
    spec_rate = spec_tokens / max(spec_dt, 1e-6)
    print(f"[spec] speculative: {spec_tokens} tokens in {spec_dt:.1f}s = {spec_rate:.0f} tok/s")
    print(f"[spec] speedup vs baseline: {spec_rate / (base_tokens / max(base_dt, 1e-6)):.2f}x "
          f"| mean tokens/step: {spec_tokens / max(steps_tot, 1):.2f}")
    print()
    print(f"{'depth':>6}  {'acceptance':>10}  {'accepted':>10}  {'attempts':>10}")
    for k in range(sd.num_mtp):
        a = acc_tot[k] / att_tot[k] if att_tot[k] else float("nan")
        print(f"{'mtp_' + str(k + 1):>6}  {a:>10.4f}  {acc_tot[k]:>10}  {att_tot[k]:>10}")


if __name__ == "__main__":
    main()
