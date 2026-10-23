"""
generate_data.py -- create MTP training data by sampling the frozen RWKV-7.

Plan (self-distillation / on-policy data for speculative decoding):
  1. Stream examples from a chosen corpus (--dataset): allenai/tulu-3-sft-mixture
     (default; filtering the broken "ai2-adapt-dev/flan_v2_converted" source) or
     the four no_think configs of openbmb/UltraData-SFT-2605 (equal share each).
  2. Render each conversation as an RWKV-7 G1 prompt, keeping the turns up to
     the final user message (the faithful assistant turn is dropped).
  3. Prefill each prompt through the frozen backbone in RNN mode, then decode a
     response in batched lockstep.
  4. Write prompt+response+[EOS] token ids to a flat uint16 .bin for train_mtp.py.

Backends (--engine)
-------------------
* v3a (default): the faster3a_2607 engine -- fused CUDA kernels, CUDA-graph
  replay for the decode step, GPU-resident token feedback (the decode loop
  runs 16 steps between CPU syncs, which matters when several worker
  processes drive GPUs concurrently on Windows), top-k/top-p GPU sampler.
  Measured end-to-end on this machine (this 2.9B model, --batch 256,
  max-new 256): RTX 4070 SUPER ~16.5 s per 256-seq block (~4.9k tok/s),
  RTX 3060 ~33 s (~2.4k tok/s), i.e. ~7.4k tok/s aggregate when the work
  queue keeps both GPUs saturated -- roughly 3.4x the legacy backend
  (~2.2k tok/s).  --v3a-deltalog adds ~3-7% decode (runs eagerly, no graph).
  NOTE: the engine uses fp16 atomics in some FFN kernels, so reruns are not
  bit-identical (occasional low-bit token flips; totals are stable).
  Prefill is sequential B=1 per prompt: RWKV is a pure RNN, so padding
  prompts to batch them would change the conditioning (and the sampled
  continuations).
* legacy: the original in-process fp16 BatchedRWKV7 (wkv7sb kernel), kept for
  reference/fallback.

Multi-GPU
---------
By default (--devices auto) one worker process is spawned per CUDA GPU.  Work
is handed out as "blocks" (batches of --batch prompts) through a shared work
queue: whichever GPU finishes its current block first pulls the next one, so a
fast GPU never idles waiting for a slow one (e.g. 4070 SUPER vs 3060 12GB).
The main process streams + tokenizes the dataset, writes finished blocks to the
output .bin in block order, and each block's sampler is seeded from the block
index -- so the dataset contents are deterministic regardless of which GPU ran
which block (bit-reproducible across identical GPUs; mixed architectures
differ in fp16 low bits and may occasionally sample different tokens).
Single-GPU runs (--devices single, or only one visible GPU) run in-process.

Quick checks:
    python scripts/generate_data.py --smoke
    python scripts/generate_data.py --num-samples 8 --max-new 64 --batch 8 \\
        --out data/tulu_smoke.bin
    # both CUDA GPUs, v3a engine, batch 256 (best measured config):
    python scripts/generate_data.py --num-samples 50000 --max-new 256 --batch 256 \\
        --out data/tulu_selfgen.bin
    # the UltraData no_think mix (Math/Code/Multi-lang-Math/IF/Knowledge):
    python scripts/generate_data.py --dataset ultradata --num-samples 50000 \\
        --max-new 256 --batch 256 --out data/ultradata_selfgen.bin
    # add DeltaLog (needs --batch 256):
    python scripts/generate_data.py --num-samples 50000 --max-new 256 --batch 256 \\
        --v3a-deltalog --out data/tulu_selfgen.bin
    # explicit device list / original backend:
    python scripts/generate_data.py --devices cuda:0,cuda:1 --num-samples 50000 ...
    python scripts/generate_data.py --engine legacy --devices single \\
        --device cuda:1 --num-samples 50000 ...
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
import math
import multiprocessing as mp
import os
import queue as pyqueue
import re
import sys
import threading
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

# Reuse mtp's build-environment fixes (CUDA_HOME, CL, -std=c++20, flag split).
# NOTE: this runs at import time in every spawned worker too -- that is exactly
# what the workers need before loading the batched CUDA kernel.
# Project root (the bootstrap at the top already put it on sys.path).  ``cuda/``
# and ``faster3a_2607/`` are resolved against the root, not against scripts/.
_THIS_DIR = _ROOT
import mtp  # noqa: E402

mtp._prepare_build_env()
from trackio_helper import TrackioRun  # noqa: E402
from torch.utils import cpp_extension as _cpp_ext  # noqa: E402

# Capture the un-patched load BEFORE mtp wraps it: the v3a engine needs its own
# flag handling (see _load_v3a below).
_CPP_LOAD_PRE_MTP = _cpp_ext.load
mtp._patch_cpp_extension_flags()

from torch.utils.cpp_extension import load  # noqa: E402

DTYPE = torch.half


# ---------------------------------------------------------------------------
# Tokenizer (from rwkv_v7_demo_fast.py, unchanged)
# ---------------------------------------------------------------------------

class RWKV_TOKENIZER:
    def __init__(self, file_name):
        self.idx2token = {}
        sorted_tokens = []
        for l in open(file_name, "r", encoding="utf-8").readlines():
            idx = int(l[: l.index(" ")])
            x = eval(l[l.index(" ") : l.rindex(" ")])
            x = x.encode("utf-8") if isinstance(x, str) else x
            sorted_tokens += [x]
            self.idx2token[idx] = x
        self.token2idx = {v: int(k) for k, v in self.idx2token.items()}

        self.table = [[[] for _ in range(256)] for _ in range(256)]
        self.good = [set() for _ in range(256)]
        self.wlen = [0 for _ in range(256)]
        for i in reversed(range(len(sorted_tokens))):
            s = sorted_tokens[i]
            if len(s) >= 2:
                s0, s1 = int(s[0]), int(s[1])
                self.table[s0][s1] += [s]
                self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)

    def encodeBytes(self, src: bytes):
        src_len, tokens, i = len(src), [], 0
        while i < src_len:
            s = src[i : i + 1]
            if i < src_len - 1:
                s1, s0 = int(src[i + 1]), int(src[i])
                if s1 in self.good[s0]:
                    sss = src[i : i + self.wlen[s0]]
                    try:
                        s = next(filter(sss.startswith, self.table[s0][s1]))
                    except StopIteration:
                        pass
            tokens.append(self.token2idx[s])
            i += len(s)
        return tokens

    def encode(self, src: str):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        return b"".join(self.idx2token[i] for i in tokens).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Batched RNN-mode RWKV-7
# ---------------------------------------------------------------------------

def _load_state(pth_path):
    """torch.load, preferring mmap so that each GPU worker maps the same .pth
    file instead of materialising its own full CPU copy (matters when 2+
    workers load the ~6 GB checkpoint at the same time)."""
    try:
        return torch.load(pth_path, map_location="cpu", weights_only=True, mmap=True)
    except (TypeError, RuntimeError):
        return torch.load(pth_path, map_location="cpu", weights_only=True)


def _load_batched_kernel(head_size: int):
    load(
        name="wkv7sb",
        sources=[os.path.join(_THIS_DIR, "cuda", "wkv7s_batched_op.cpp"),
                 os.path.join(_THIS_DIR, "cuda", "wkv7s_batched.cu")],
        is_python_module=False,
        verbose=False,
        extra_cuda_cflags=[
            "-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3",
            "--extra-device-vectorization", f"-D_N_={head_size}",
        ],
    )


class BatchedRWKV7:
    """RNN-mode RWKV-7 with a real batch dimension (fp16, inference only)."""

    def __init__(self, pth_path, n_layer, n_embd, head_size, device="cuda:0",
                 keep_unfused: bool = False):
        self.device = device
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.head_size = head_size

        self.z = _load_state(pth_path)
        n_head, hs = self.z["blocks.0.att.r_k"].shape
        assert hs == head_size, (hs, head_size)
        self.n_head = n_head

        for k in list(self.z.keys()):
            v = self.z[k]
            if any(s in k for s in (".key.weight", ".value.weight", ".receptance.weight",
                                    ".output.weight", "head.weight")):
                v = v.t()
            v = v.squeeze()
            self.z[k] = v
            if k.endswith("att.r_k"):
                self.z[k] = v.reshape(-1)

        # fuse the layer-0 embedding norm into the embedding (as the demos do);
        # the raw embedding is only needed by the MTP heads, so keep it only on
        # request (it costs ~n_vocab*n_embd*2 bytes, e.g. 335 MB here).
        self.emb_unfused = (self.z["emb.weight"].to(device=device, dtype=DTYPE)
                            if keep_unfused else None)
        emb = self.z["emb.weight"].float()
        self.z["emb.weight"] = F.layer_norm(
            emb, (n_embd,),
            weight=self.z["blocks.0.ln0.weight"].float(),
            bias=self.z["blocks.0.ln0.bias"].float(),
        ).to(DTYPE)
        # layer 0 value-residual weights are unused; point them at a* (as demos do)
        self.z["blocks.0.att.v0"] = self.z["blocks.0.att.a0"]
        self.z["blocks.0.att.v1"] = self.z["blocks.0.att.a1"]
        self.z["blocks.0.att.v2"] = self.z["blocks.0.att.a2"]

        for k in self.z:
            self.z[k] = self.z[k].to(device=device, dtype=DTYPE)

        _load_batched_kernel(head_size)

    def new_state(self, batch: int):
        C, H, N = self.n_embd, self.n_head, self.head_size
        st = []
        for _ in range(self.n_layer):
            st.append([
                torch.zeros(batch, C, dtype=DTYPE, device=self.device),
                torch.zeros(batch, H, N, N, dtype=torch.float32, device=self.device),
                torch.zeros(batch, C, dtype=DTYPE, device=self.device),
            ])
        return st

    @staticmethod
    def stack_states(states):
        n_layer = len(states[0])
        out = []
        for l in range(n_layer):
            out.append([torch.cat([s[l][c] for s in states], dim=0) for c in range(3)])
        return out

    @torch.no_grad()
    def prefill_batch(self, batch_prompts):
        """Prefill B prompts into ONE preallocated batch state.

        Numerically identical to ``[forward(p, None) for p in prompts]`` followed
        by ``stack_states`` (the same float values land in the same layout), but
        it never holds the B single-sequence states and the stacked copy at the
        same time.  Peak drops from ``2*B`` to ``B+1`` state tensors.
        """
        B = len(batch_prompts)
        state = self.new_state(B)
        last_logits = []
        for i, ids in enumerate(batch_prompts):
            t = torch.tensor(ids, device=self.device).unsqueeze(0)
            logits, st = self.forward(t, None)
            for l in range(self.n_layer):
                state[l][0][i].copy_(st[l][0][0])
                state[l][1][i].copy_(st[l][1][0])
                state[l][2][i].copy_(st[l][2][0])
            last_logits.append(logits[:, -1, :].float())
            del st
        return state, torch.cat(last_logits, dim=0)

    @torch.no_grad()
    def forward(self, idx: torch.Tensor, state, return_hidden: bool = False,
                inplace: bool = False, return_taps: bool = False):
        """idx: (B, T).  Returns logits (B, T, V) and state; if return_hidden
        also returns the post-ln_out hidden states (B, T, C).

        return_taps=True additionally returns the EAGLE-3 mid-backbone tap
        states ``(x_early, x_mid)`` (each (B, T, C), raw residual stream,
        captured after layers ``n_layer//8-1`` and ``n_layer//2-1``), matching
        mtp.RWKVWithMTP._forward_features_multi.  The final tap is the returned
        post-ln_out hidden itself.  With return_taps the success tuple grows by
        one element: (hidden, logits, state, taps).

        inplace=True reuses the passed state (no clone) -- safe for plain
        generation, but NOT for speculative verification, which must keep the
        prefix state.
        """
        z = self.z
        B, T = idx.shape
        C, H, N = self.n_embd, self.n_head, self.head_size
        if state is None:
            state = self.new_state(B)

        i_early = max(1, self.n_layer // 8 - 1)
        i_mid = self.n_layer // 2 - 1

        x = z["emb.weight"][idx]
        v_first = None
        x_early = x_mid = None
        state_out = state if inplace else []

        for i in range(self.n_layer):
            att = f"blocks.{i}.att."
            ffn = f"blocks.{i}.ffn."
            bbb = f"blocks.{i}."

            xn = F.layer_norm(x, (C,), z[bbb + "ln1.weight"], z[bbb + "ln1.bias"])
            x_prev = state[i][0]
            xp = x_prev.unsqueeze(1) if T == 1 else torch.cat(
                [x_prev.unsqueeze(1), xn[:, :-1, :]], dim=1)
            d = xp - xn
            xr = xn + d * z[att + "x_r"]
            xw = xn + d * z[att + "x_w"]
            xk = xn + d * z[att + "x_k"]
            xv = xn + d * z[att + "x_v"]
            xa = xn + d * z[att + "x_a"]
            xg = xn + d * z[att + "x_g"]

            r = xr @ z[att + "receptance.weight"]
            w = torch.tanh(xw @ z[att + "w1"]) @ z[att + "w2"]
            k = xk @ z[att + "key.weight"]
            v = xv @ z[att + "value.weight"]
            a = torch.sigmoid(z[att + "a0"] + (xa @ z[att + "a1"]) @ z[att + "a2"])
            g = torch.sigmoid(xg @ z[att + "g1"]) @ z[att + "g2"]

            kk = F.normalize((k * z[att + "k_k"]).view(B, T, H, N), dim=-1, p=2.0).view(B, T, C)
            k = k * (1 + (a - 1) * z[att + "k_a"])
            if i == 0:
                v_first = v
            else:
                v = v + (v_first - v) * torch.sigmoid(
                    z[att + "v0"] + (xv @ z[att + "v1"]) @ z[att + "v2"])

            w_pre = -F.softplus(-(z[att + "w0"] + w)) - 0.5
            y = torch.empty((B, T, C), device=self.device, dtype=DTYPE)
            kv_out = state[i][1] if inplace else state[i][1].clone()
            torch.ops.wkv7sb.forward(
                B, T, C, H, kv_out,
                r.contiguous(), w_pre.contiguous(), k.contiguous(), v.contiguous(),
                (-kk).contiguous(), (kk * a).contiguous(), y)

            y = F.group_norm(y.view(B * T, C), num_groups=H,
                             weight=z[att + "ln_x.weight"], bias=z[att + "ln_x.bias"],
                             eps=64e-5).view(B, T, H, N)
            y = y + (r.view(B, T, H, N) * k.view(B, T, H, N) * z[att + "r_k"].view(1, 1, H, N)
                     ).sum(dim=-1, keepdim=True) * v.view(B, T, H, N)
            y = (y.view(B, T, C) * g) @ z[att + "output.weight"]
            x = x + y

            xn2 = F.layer_norm(x, (C,), z[bbb + "ln2.weight"], z[bbb + "ln2.bias"])
            xp2 = state[i][2].unsqueeze(1) if T == 1 else torch.cat(
                [state[i][2].unsqueeze(1), xn2[:, :-1, :]], dim=1)
            d2 = xp2 - xn2
            k2 = xn2 + d2 * z[ffn + "x_k"]
            k2 = torch.relu(k2 @ z[ffn + "key.weight"]) ** 2
            x = x + k2 @ z[ffn + "value.weight"]
            if inplace:
                state[i][0] = xn[:, -1, :]
                state[i][2] = xn2[:, -1, :]
            else:
                state_out.append([xn[:, -1, :], kv_out, xn2[:, -1, :]])
            if i == i_early:
                x_early = x
            elif i == i_mid:
                x_mid = x

        x = F.layer_norm(x, (C,), z["ln_out.weight"], z["ln_out.bias"])
        logits = x @ z["head.weight"]
        if return_hidden:
            if return_taps:
                return x, logits, state_out, (x_early, x_mid)
            return x, logits, state_out
        return logits, state_out


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_batch(logits: torch.Tensor, temperature=1.0, top_p=0.0, top_k=0,
                 generator=None) -> torch.Tensor:
    """logits: (B, V) float.  Returns (B,) token ids."""
    if temperature <= 0:
        return logits.argmax(dim=-1)
    probs = F.softmax(logits / temperature, dim=-1)
    if top_k and top_k > 0:
        kth = torch.topk(probs, min(top_k, probs.size(-1)), dim=-1).values[:, -1:]
        probs = torch.where(probs < kth, torch.zeros_like(probs), probs)
    if top_p and 0 < top_p < 1:
        sp, si = torch.sort(probs, descending=True, dim=-1)
        cum = torch.cumsum(sp, dim=-1)
        drop = cum > top_p
        drop[..., 1:] = drop[..., :-1].clone()
        drop[..., 0] = False
        sp = torch.where(drop, torch.zeros_like(sp), sp)
        probs = torch.zeros_like(probs).scatter_(-1, si, sp)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


# ---------------------------------------------------------------------------
# tulu-3-sft-mixture -> G1 prompt
# ---------------------------------------------------------------------------

def clean_txt(txt: str) -> str:
    return re.sub(r"\n{2,}", "\n", (txt or "").replace("\r\n", "\n")).strip()


def build_prompt(messages) -> str:
    msgs = list(messages)
    if msgs and msgs[-1].get("role") == "assistant":
        msgs = msgs[:-1]  # drop the faithful answer; we generate our own
    parts = []
    for m in msgs:
        role = m.get("role", "user")
        content = clean_txt(m.get("content", ""))
        if not content:
            continue
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(f"User: {content}")
    if not parts:
        return ""
    prompt = "\n\n".join(parts)
    if not prompt.endswith("Assistant:"):
        prompt += "\n\nAssistant:"
    return prompt


def iter_tulu(num_samples: int, seed: int):
    """Stream allenai/tulu-3-sft-mixture, skipping the broken flan_v2 source."""
    from datasets import load_dataset
    ds = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10000)
    kept = 0
    for ex in ds:
        if ex.get("source") == "ai2-adapt-dev/flan_v2_converted":
            continue
        yield ex
        kept += 1
        if kept >= num_samples:
            return


# openbmb/UltraData-SFT-2605 (--dataset ultradata): the four no_think configs.
ULTRADATA_DATASET = "openbmb/UltraData-SFT-2605"
# All read from their "no_think" split; Multi-lang-Math only ships a no_think
# split, and the other available configs (Chinese-general, Multi-lang-Knowledge)
# are intentionally ignored.
ULTRADATA_CONFIGS = ("Math", "Code", "Multi-lang-Math", "IF", "Knowledge")


def _open_ultradata_stream(config: str, seed: int):
    """Open a streaming iterator over one config's no_think split.

    Streaming fetches shards lazily; iterating stops as soon as the per-config
    quota is reached, so the full dataset is never downloaded.  The bounded
    shuffle only reorders a small window of samples (no materialisation).
    """
    from datasets import load_dataset
    ds = load_dataset(ULTRADATA_DATASET, config, split="no_think", streaming=True)
    return iter(ds.shuffle(seed=seed, buffer_size=10000))


def _per_config_quotas(num_samples: int):
    """Split num_samples as evenly as possible across ULTRADATA_CONFIGS
    (e.g. 10000 -> 2500 each); any remainder goes to the first configs."""
    base, rem = divmod(num_samples, len(ULTRADATA_CONFIGS))
    return [base + (1 if i < rem else 0) for i in range(len(ULTRADATA_CONFIGS))]


# ---------------------------------------------------------------------------
# Work units: one block = one batch of prompts (the unit handed to GPUs)
# ---------------------------------------------------------------------------

def _encode_prompt(args, tokenizer, prompt):
    """Render + tokenize one prompt; None if it is unusable."""
    if not prompt:
        return None
    ids = tokenizer.encode(prompt)
    if len(ids) < 4:
        return None
    return ids[-args.max_prompt:]


def _prompt_stream_tulu(args, tokenizer):
    """Yield blocks of prompt token-id lists from tulu-3-sft-mixture.  The
    final block may hold fewer than --batch prompts.  Deterministic."""
    buffer = []
    for ex in iter_tulu(args.num_samples, args.seed):
        ids = _encode_prompt(args, tokenizer, build_prompt(ex.get("messages", [])))
        if ids is None:
            continue
        buffer.append(ids)
        if len(buffer) >= args.batch:
            yield buffer
            buffer = []
    if buffer:
        yield buffer


def _prompt_stream_ultradata(args, tokenizer):
    """Yield blocks drawn round-robin from the UltraData no_think configs, so
    every block is evenly mixed and each config contributes its quota.  The
    final block may hold fewer than --batch prompts.  Deterministic."""
    quotas = _per_config_quotas(args.num_samples)
    streams = [_open_ultradata_stream(cfg, args.seed) for cfg in ULTRADATA_CONFIGS]
    taken = [0] * len(ULTRADATA_CONFIGS)
    active = [q > 0 for q in quotas]
    buffer = []

    while any(active):
        for i in range(len(ULTRADATA_CONFIGS)):
            if not active[i]:
                continue
            # Pull from config i until a usable prompt appears or it runs out.
            ids = None
            while True:
                try:
                    ex = next(streams[i])
                except StopIteration:
                    break
                ids = _encode_prompt(args, tokenizer,
                                     build_prompt(ex.get("messages", [])))
                if ids is not None:
                    break
            if ids is None:  # exhausted before reaching the quota
                active[i] = False
                continue
            buffer.append(ids)
            taken[i] += 1
            if taken[i] >= quotas[i]:
                active[i] = False
            if len(buffer) >= args.batch:
                yield buffer
                buffer = []
    if buffer:
        yield buffer


# Dispatch on --dataset.  Both paths are deterministic: same seed -> same blocks.
_PROMPT_STREAMS = {
    "tulu": _prompt_stream_tulu,
    "ultradata": _prompt_stream_ultradata,
}


def _prompt_stream(args, tokenizer):
    return _PROMPT_STREAMS[args.dataset](args, tokenizer)


def _generate_block(model, prompts, *, max_new, temperature, top_p, top_k,
                    generator):
    """Prefill + batched lockstep decode of one block.

    Returns (flat uint16 array with prompt+response+[EOS] per row,
    n_seq, n_tok).  Frees the block state before returning.
    """
    B = len(prompts)
    state, logits = model.prefill_batch(prompts)  # (B,V) last-token logits

    gen_ids = [[] for _ in range(B)]
    done = [False] * B
    for _ in range(max_new):
        tok = sample_batch(logits, temperature, top_p, top_k, generator)
        for b in range(B):
            if not done[b]:
                gen_ids[b].append(int(tok[b]))
                if int(tok[b]) == 0:
                    done[b] = True
        if all(done):
            break
        feed = tok.clone()
        if any(done):
            feed[done] = 0
        lg, state = model.forward(feed.view(B, 1), state, inplace=True)
        logits = lg[:, -1, :].float()

    seqs = [np.asarray(p + g + [0], dtype=np.uint16)
            for p, g in zip(prompts, gen_ids)]
    arr = np.concatenate(seqs) if seqs else np.zeros(0, dtype=np.uint16)
    n_tok = int(arr.size)
    del state, logits, seqs
    torch.cuda.empty_cache()
    return arr, B, n_tok


# ---------------------------------------------------------------------------
# Optional fast engine backend: faster3a_2607/rwkv7_fast_v3a.py
# ---------------------------------------------------------------------------

_V3A_DIR = os.path.join(_THIS_DIR, "faster3a_2607")


def _make_v3a_load():
    """Build a cpp_extension.load for the v3a engine's kernels.

    Differences vs the mtp-patched load used for cuda/wkv7s_batched.*:
    * NO unconditional -std=c++20: nvcc 13.2 misparses rwkv7_v3a_ops.cu's
      ``x.data_ptr<dtype>()`` under -std=c++20 (plain torch -std=c++17 builds
      fine). Only rwkv7_wkv_fp32_v2.cu needs -std=c++20 (torch 2.10/MSVC
      C2872 'std': ambiguous).
    * explicit cublas.lib/cublasLt.lib linking on Windows: the engine calls
      cublasLt directly, which links transitively on Linux but must be named
      explicitly for MSVC.
    """
    base = _CPP_LOAD_PRE_MTP
    is_win = os.name == "nt"
    cuda_lib = os.path.join(os.environ.get("CUDA_HOME", ""), "lib", "x64")

    def _split(flags):
        out = []
        for f in flags or ():
            if isinstance(f, str) and " " in f:
                out.extend(f.split())
            else:
                out.append(f)
        return out

    def load(*args, **kwargs):
        for key in ("extra_cuda_cflags", "extra_cflags", "extra_ldflags"):
            if key in kwargs:
                kwargs[key] = _split(kwargs[key])
        name = args[0] if args else kwargs.get("name", "")
        cc = list(kwargs.get("extra_cuda_cflags") or [])
        if "fp32" in str(name) and "-std=c++20" not in cc:
            cc.append("-std=c++20")
        kwargs["extra_cuda_cflags"] = cc
        if is_win:
            ld = list(kwargs.get("extra_ldflags") or [])
            for lib in ("cublas.lib", "cublasLt.lib"):
                p = os.path.join(cuda_lib, lib)
                if p not in ld:
                    ld.append(p)
            kwargs["extra_ldflags"] = ld
        return base(*args, **kwargs)

    return load


def _load_v3a(wkv_mode: str):
    """Import rwkv7_fast_v3a and build/load its CUDA extensions."""
    if not os.path.isdir(_V3A_DIR):
        raise RuntimeError(
            f"--engine v3a requires the engine at {_V3A_DIR} (not found)")
    if _V3A_DIR not in sys.path:
        sys.path.insert(0, _V3A_DIR)
    # Swap the module-level load BEFORE importing the engine: its
    # ``from torch.utils.cpp_extension import load`` must bind our wrapper.
    saved = _cpp_ext.load
    _cpp_ext.load = _make_v3a_load()
    try:
        import rwkv7_fast_v3a as v3a
        v3a.load_extensions(wkv_mode)
    finally:
        _cpp_ext.load = saved
    return v3a


def _v3a_sample(logits, temperature, top_p, top_k, generator):
    """top-k + top-p sampler over the engine's (B, V) logits (GPU, no 65k sort).

    Same distribution as app4b's sample_logits_batch_cuda; the optional
    ``generator`` keeps per-block sampling reproducible.
    """
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)
    k = top_k if 0 < top_k <= logits.size(-1) else 512
    vals, ids = torch.topk(logits.float(), k=k, dim=-1, sorted=True)
    probs = torch.softmax(vals / temperature, dim=-1)
    cdf = torch.cumsum(probs, dim=-1)
    if 0 < top_p < 1:
        keep = torch.argmax((cdf >= top_p).to(torch.int32), dim=-1)
        mass = cdf.gather(1, keep.view(-1, 1))
    else:
        mass = cdf[:, -1:]
    r = torch.rand((logits.size(0), 1), device=logits.device,
                   generator=generator) * mass
    out = torch.searchsorted(cdf, r.contiguous()).clamp_max(k - 1)
    return ids.gather(1, out).view(-1)


class V3aRunner:
    """Adapter around rwkv7_fast_v3a.RWKV7 for one worker process.

    Fixed decode batch ``B``; the decode step (embed buffer -> forward_from_x
    -> last-token logits) is captured into a CUDA graph once and replayed per
    token, matching the engine's own app4b demo.  Prefill stays sequential
    B=1: RWKV is a pure RNN, so padding prompts to batch them would change
    the states (and thus the sampled continuations) -- each prompt must be
    conditioned on exactly its own tokens.
    """

    def __init__(self, pth_path, wkv_mode="fp16", batch=256, deltalog=False,
                 max_new=256):
        self.v3a = _load_v3a(wkv_mode)
        # Embedding must live on-GPU: the decode graph reads the next-step
        # token ids from a device buffer (the decode loop runs K steps between
        # CPU syncs, so the CPU cannot feed embeddings each step).
        self.v3a.EMB_DEVICE = "gpu"
        self.v3a.MODEL_PATH = pth_path
        self.model = self.v3a.RWKV7()
        self.wkv_mode = wkv_mode
        self.C = self.v3a.C
        self.V = self.v3a.V
        self.L = self.v3a.L
        self.decode_batch = batch
        self.deltalog = deltalog
        self.max_new = max_new
        self.deltalog_m = 0
        if deltalog:
            if wkv_mode != "fp16":
                raise RuntimeError("--v3a-deltalog requires --v3a-wkv fp16")
            self.deltalog_m = self.v3a.WKV_DELTALOG_TUNED_M.get(
                (self.C, batch), 0)
            if not self.deltalog_m:
                tuned = sorted(b for (c, b) in self.v3a.WKV_DELTALOG_TUNED_M
                               if c == self.C)
                raise RuntimeError(
                    f"no tuned DeltaLog path for C={self.C} batch={batch}; "
                    f"use --batch one of {tuned} or drop --v3a-deltalog")
        self.dec = None
        self.logits_buf = None
        self._state1 = None
        self._phase = 0  # DeltaLog cycle phase of the decode state

    def _decode_ctx(self):
        if self.dec is not None:
            return self.dec
        m, v3a = self.model, self.v3a
        B = self.decode_batch
        state = m.zero_state(B)
        x = torch.empty((B, 1, self.C), dtype=torch.half, device="cuda")
        path = v3a.select_path(B, 1)
        # GPU-resident decode-loop plumbing (see decode()): the captured graph
        # consumes idx_buf (next token per row); done rows keep being fed EOS.
        self.idx_buf = torch.zeros(B, dtype=torch.long, device="cuda")
        self.zeros_b = torch.zeros(B, dtype=torch.long, device="cuda")
        self.done_gpu = torch.zeros(B, dtype=torch.bool, device="cuda")
        self.tok_hist = torch.zeros((max(self.max_new, 1), B),
                                    dtype=torch.long, device="cuda")
        graph = None
        if not self.deltalog:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    m.forward(self.idx_buf.view(B, 1), state)
            torch.cuda.current_stream().wait_stream(stream)
            try:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    out = m.forward(self.idx_buf.view(B, 1), state)
                torch.cuda.synchronize()
            except Exception as exc:
                print(f"[gen v3a] CUDA graph disabled for B={B}: {exc}",
                      flush=True)
                graph = None
                out = None
        if graph is None:
            out = torch.empty((B, self.V), dtype=torch.half, device="cuda")
        self.logits_buf = torch.empty((B, self.V), dtype=torch.float32,
                                      device="cuda")
        self._state1 = m.zero_state(1)
        self.dec = (B, state, x, path, graph, out)
        return self.dec

    def _advance_cycle(self, steps):
        """Run `steps` DeltaLog phases with EOS embeddings to reach a cycle
        boundary (state content becomes garbage; the next block overwrites it)."""
        B, state, x, path, graph, out = self._decode_ctx()
        for _ in range(steps):
            self.model.forward_deltalog_step(
                torch.zeros((B, 1), dtype=torch.long), state,
                merge_interval=self.deltalog_m)
            self._phase = (self._phase + 1) % self.deltalog_m

    def start_block(self, prompts):
        """Prefill len(prompts) prompts into the fixed batch state."""
        B, state, x, path, graph, out = self._decode_ctx()
        m = self.model
        if self.deltalog and self._phase:
            # finish the previous block's cycle so the log workspace is empty
            # before the base state gets overwritten by the new prefill
            self._advance_cycle(self.deltalog_m - self._phase)
        self.done_gpu.zero_()
        self.idx_buf.zero_()
        st1 = self._state1
        st1[0].zero_()
        st1[1].zero_()
        st1[2].zero_()
        for i, p in enumerate(prompts):
            ids = torch.tensor(p, dtype=torch.long).view(1, -1)
            logits1 = m.forward(ids, st1)  # (1, V) last-token logits
            for l in range(self.L):
                state[0][l][:, i].copy_(st1[0][l][:, 0])
                state[1][l][i].copy_(st1[1][l][0])
            state[2][i] = st1[2][0]
            self.logits_buf[i] = logits1[0]
            st1[0].zero_()
            st1[1].zero_()
            st1[2].zero_()

    def decode(self, *, max_new, temperature, top_p, top_k, generator):
        """Batched lockstep decode from the prefilled state; returns per-row
        generated token id lists (EOS-terminated or max_new long).

        Runs SYNC_EVERY steps between CPU syncs: per step the GPU replays the
        captured graph (embed(idx_buf) -> forward -> logits), samples eagerly
        on-device, and updates idx_buf/done_gpu on its own -- the CPU only
        harvests tok_hist windows.  This keeps per-step CPU work (and WDDM
        cross-process submissions) near zero, which matters when several
        worker processes drive GPUs concurrently on Windows.
        """
        SYNC_EVERY = 16
        B, state, x, path, graph, out = self._decode_ctx()
        m = self.model
        logits = self.logits_buf
        gen_ids = [[] for _ in range(B)]
        done = [False] * B
        step = 0
        while step < max_new:
            k = min(SYNC_EVERY, max_new - step)
            for j in range(k):
                if graph is not None:
                    graph.replay()
                    out_logits = out
                elif self.deltalog:
                    out_logits = m.forward_deltalog_step(
                        self.idx_buf.view(B, 1), state,
                        merge_interval=self.deltalog_m)
                    self._phase = (self._phase + 1) % self.deltalog_m
                else:
                    out_logits = m.forward_from_x(
                        m.embed(self.idx_buf.view(B, 1)), state, path)
                tok = _v3a_sample(out_logits, temperature, top_p, top_k,
                                  generator)
                self.done_gpu |= tok == 0
                self.idx_buf.copy_(torch.where(self.done_gpu, self.zeros_b, tok))
                self.tok_hist[step + j].copy_(tok)
            # one sync per window: harvest the generated tokens
            hist = self.tok_hist[step:step + k].tolist()
            active = 0
            for j in range(k):
                row = hist[j]
                for b in range(B):
                    if done[b]:
                        continue
                    t = row[b]
                    gen_ids[b].append(t)
                    if t == 0:
                        done[b] = True
                    else:
                        active += 1
            step += k
            if active == 0 or step >= max_new:
                break
        return gen_ids


def _generate_block_v3a(runner, prompts, args, bidx):
    """One block on the v3a engine.  Returns (flat uint16 array, n_seq, n_tok)."""
    B_dec = runner.decode_batch
    n = len(prompts)
    work = prompts if n == B_dec else prompts + [prompts[0]] * (B_dec - n)
    gen = torch.Generator(device="cuda").manual_seed(
        (args.seed + 1_000_003 * bidx) % (2 ** 31 - 1))
    runner.start_block(work)
    gen_ids = runner.decode(max_new=args.max_new, temperature=args.temperature,
                            top_p=args.top_p, top_k=args.top_k,
                            generator=gen)[:n]
    seqs = [np.asarray(p + g + [0], dtype=np.uint16)
            for p, g in zip(prompts, gen_ids)]
    arr = np.concatenate(seqs) if seqs else np.zeros(0, dtype=np.uint16)
    torch.cuda.synchronize()
    return arr, n, int(arr.size)


# ---------------------------------------------------------------------------
# Multi-GPU: one worker process per GPU pulling blocks off a shared queue
# ---------------------------------------------------------------------------

def _queue_finish(q):
    """Flush + close a queue handle from the worker side so a final put cannot
    be lost when the process exits right away."""
    try:
        q.close()
        q.join_thread()
    except Exception:
        pass


def _gpu_worker(rank, device, cfg, args, block_q, res_q):
    """Worker loop: pull (block_idx, prompts) off the shared work queue,
    generate on this GPU, push results back.  Exits after a None sentinel."""
    use_v3a = args.engine == "v3a"
    try:
        if use_v3a:
            # CUDA_VISIBLE_DEVICES was inherited per-process from the parent;
            # the single visible GPU is cuda:0 here.
            torch.cuda.set_device(0)
            print(f"[gen {device}] v3a engine ({args.v3a_wkv}) loading "
                  f"{args.pth} (batch {args.batch}) on "
                  f"{torch.cuda.get_device_name(0)}...", flush=True)
            runner = V3aRunner(args.pth, wkv_mode=args.v3a_wkv,
                               batch=args.batch, deltalog=args.v3a_deltalog,
                               max_new=args.max_new)
            print(f"[gen {device}] v3a ready", flush=True)
        else:
            torch.cuda.set_device(torch.device(device))
            print(f"[gen {device}] loading model {args.pth} "
                  f"(L{cfg['num_hidden_layers']} D{cfg['hidden_size']}, fp16)...",
                  flush=True)
            model = BatchedRWKV7(args.pth, cfg["num_hidden_layers"],
                                 cfg["hidden_size"], cfg["head_dim"], device=device)
            print(f"[gen {device}] ready", flush=True)
    except Exception:
        res_q.put(("fatal", rank, device, traceback.format_exc()))
        _queue_finish(res_q)
        return

    while True:
        item = block_q.get()
        if item is None:
            break
        bidx, prompts_np = item
        prompts = [p.tolist() for p in prompts_np]
        t0 = time.time()
        try:
            # per-block seed => the generated data does not depend on which
            # GPU (or in which order) blocks were processed
            if use_v3a:
                arr, bseq, btok = _generate_block_v3a(runner, prompts, args, bidx)
            else:
                gen = torch.Generator(device=device).manual_seed(
                    (args.seed + 1_000_003 * bidx) % (2 ** 31 - 1))
                arr, bseq, btok = _generate_block(
                    model, prompts, max_new=args.max_new,
                    temperature=args.temperature, top_p=args.top_p,
                    top_k=args.top_k, generator=gen)
        except torch.cuda.OutOfMemoryError:
            res_q.put(("error", bidx, device,
                       f"CUDA OOM on block {bidx} (batch {len(prompts)}). "
                       f"Lower --batch.\n{traceback.format_exc()}"))
            _queue_finish(res_q)
            return
        except Exception:
            res_q.put(("error", bidx, device, traceback.format_exc()))
            _queue_finish(res_q)
            return
        res_q.put(("ok", bidx, device, arr, bseq, btok, time.time() - t0))
    _queue_finish(res_q)


def _run_multi(args, cfg, tokenizer, devices, tracker: TrackioRun | None = None):
    n_workers = len(devices)
    n_blocks_expected = math.ceil(args.num_samples / args.batch)

    ctx = mp.get_context("spawn")
    block_q = ctx.Queue(maxsize=2 * n_workers + 2)  # bounded: backpressure
    res_q = ctx.Queue()

    procs = []
    for i, d in enumerate(devices):
        if args.engine == "v3a":
            # Set BEFORE this child's start(): the spawned child snapshots the
            # parent's environment at creation time, so the v3a engine's
            # hardcoded cuda:0 refers to this worker's physical GPU.
            os.environ["CUDA_VISIBLE_DEVICES"] = d.split(":")[-1]
            procs.append(ctx.Process(target=_gpu_worker, name=f"genworker-{d}",
                                     args=(i, d, cfg, args, block_q, res_q),
                                     daemon=True))
            procs[-1].start()
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            procs.append(ctx.Process(target=_gpu_worker, name=f"genworker-{d}",
                                     args=(i, d, cfg, args, block_q, res_q),
                                     daemon=True))
            procs[-1].start()

    # Producer thread: stream + tokenize the dataset, feed the work queue.
    stop = threading.Event()
    prod = {"done": False, "error": None, "blocks": 0}

    def _put(item):
        while not stop.is_set():
            try:
                block_q.put(item, timeout=1.0)
                return True
            except pyqueue.Full:
                continue
        return False

    def producer():
        try:
            for bidx, block in enumerate(_prompt_stream(args, tokenizer)):
                payload = [np.asarray(ids, dtype=np.uint16) for ids in block]
                if not _put((bidx, payload)):
                    return
                prod["blocks"] = bidx + 1
            for _ in procs:  # one exit sentinel per worker
                if not _put(None):
                    return
        except Exception:
            prod["error"] = traceback.format_exc()
            stop.set()
        finally:
            prod["done"] = True

    print(f"[gen] multi-GPU: {n_workers} workers on {', '.join(devices)}, "
          f"shared work queue, blocks of {args.batch} prompts")
    prod_t = threading.Thread(target=producer, name="producer", daemon=True)
    prod_t.start()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    t0 = time.time()
    out_f = open(args.out, "wb")
    pending = {}          # block_idx -> (arr, n_seq, n_tok) finished out of order
    next_write = 0        # next block idx allowed to hit disk (in-order output)
    n_seq = n_tok = 0
    blocks_done = 0
    per_dev = {d: {"blocks": 0, "seconds": 0.0, "tokens": 0} for d in devices}
    err = None
    last_log = 0.0

    try:
        while True:
            if prod["error"]:
                err = f"dataset streaming failed:\n{prod['error']}"
                break
            try:
                msg = res_q.get(timeout=1.0)
            except pyqueue.Empty:
                msg = None
            if msg is not None:
                kind = msg[0]
                if kind == "ok":
                    _, bidx, dev, arr, bseq, btok, dt = msg
                    pending[bidx] = (arr, bseq, btok)
                    pd = per_dev[dev]
                    pd["blocks"] += 1
                    pd["seconds"] += dt
                    pd["tokens"] += btok
                elif kind == "fatal":
                    err = f"worker {msg[2]} failed to start:\n{msg[3]}"
                    break
                else:  # "error"
                    err = f"worker {msg[2]} failed on block {msg[1]}:\n{msg[3]}"
                    break

            # write finished blocks to disk in block order (deterministic file)
            while next_write in pending:
                arr, bseq, btok = pending.pop(next_write)
                arr.tofile(out_f)
                n_seq += bseq
                n_tok += btok
                next_write += 1
                blocks_done += 1
                now = time.time()
                if blocks_done % args.log_every == 0 or now - last_log > 30:
                    dt = now - t0
                    eta = dt / blocks_done * max(n_blocks_expected - blocks_done, 0)
                    stat = " ".join(f"{d}:{pd['blocks']}" for d, pd in per_dev.items())
                    print(f"[gen] {blocks_done}/{n_blocks_expected} blocks ({stat}) | "
                          f"{n_seq} seqs, {n_tok:,} tok | "
                          f"{n_tok / max(dt, 1e-6):.0f} tok/s | ETA {eta / 60:.1f} min",
                          flush=True)
                    last_log = now
                    if tracker is not None:
                        tracker.log_now({
                            "blocks": blocks_done,
                            "blocks_total": n_blocks_expected,
                            "seqs": n_seq,
                            "tokens": n_tok,
                            "tok_s": n_tok / max(dt, 1e-6),
                            "eta_min": eta / 60.0,
                            **{f"blocks_{d.split(':')[-1]}": pd["blocks"]
                               for d, pd in per_dev.items()},
                        })

            alive = [p for p in procs if p.is_alive()]
            if prod["done"] and not alive and res_q.empty():
                if pending:  # workers died before finishing every block
                    err = ("workers exited before completing blocks "
                           f"{sorted(pending)} (likely OOM -- lower --batch)")
                break
            if not alive and not prod["done"]:
                err = "all workers exited unexpectedly"
                stop.set()
                break
    except KeyboardInterrupt:
        print("\n[gen] interrupted -- shutting down workers", flush=True)
        err = "interrupted"
    finally:
        stop.set()
        out_f.close()
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=5)
        prod_t.join(timeout=5)
        block_q.close()
        block_q.cancel_join_thread()
        res_q.close()
        res_q.cancel_join_thread()

    if err:
        print(f"[gen] ERROR: {err}", file=sys.stderr)
        if os.path.exists(args.out):
            print(f"[gen] partial output left at {args.out} "
                  "(delete it before retrying)", file=sys.stderr)
        sys.exit(1)

    for d, pd in per_dev.items():
        avg = pd["seconds"] / max(pd["blocks"], 1)
        print(f"[gen] {d}: {pd['blocks']} blocks, {pd['tokens']:,} tok, "
              f"{pd['seconds']:.0f}s ({avg:.1f}s/block avg)")
    _write_meta(args, n_seq, n_tok, t0,
                extra={"devices": devices, "per_device": per_dev,
                       "engine": args.engine})
    if tracker is not None:
        tracker.log_now({
            "done": 1,
            "seqs": n_seq,
            "tokens": n_tok,
            "minutes": (time.time() - t0) / 60.0,
        })
        tracker.finish()


# ---------------------------------------------------------------------------
# Single-GPU (original in-process behaviour, identical RNG stream)
# ---------------------------------------------------------------------------

def _run_single(args, cfg, tokenizer, tracker: TrackioRun | None = None):
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    if args.engine == "v3a":
        # single process: pin this process to the requested physical GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":")[-1]
        print(f"[gen] v3a engine ({args.v3a_wkv}) on {args.device}, "
              f"batch {args.batch}")
        runner = V3aRunner(args.pth, wkv_mode=args.v3a_wkv,
                           batch=args.batch, deltalog=args.v3a_deltalog,
                           max_new=args.max_new)
        n_seq = n_tok = 0
        t0 = time.time()
        buffer_out = []
        for bidx, block in enumerate(_prompt_stream(args, tokenizer)):
            arr, bseq, btok = _generate_block_v3a(runner, block, args, bidx)
            buffer_out.append(arr)
            n_seq += bseq
            n_tok += btok
            if n_seq % (args.batch * args.log_every) == 0:
                dt = time.time() - t0
                print(f"[gen] {n_seq} seqs, {n_tok:,} tokens, "
                      f"{n_tok / max(dt, 1e-6):.0f} tok/s, {dt / 60:.1f} min")
                if tracker is not None:
                    tracker.log_now({"seqs": n_seq, "tokens": n_tok,
                                     "tok_s": n_tok / max(dt, 1e-6),
                                     "minutes": dt / 60.0})
        arr = np.concatenate(buffer_out) if buffer_out else np.zeros(0, dtype=np.uint16)
        arr.tofile(args.out)
        _write_meta(args, n_seq, int(arr.size), t0,
                    extra={"devices": [args.device], "engine": "v3a"})
        if tracker is not None:
            tracker.log_now({"done": 1, "seqs": n_seq, "tokens": int(arr.size),
                             "minutes": (time.time() - t0) / 60.0})
            tracker.finish()
        return

    print(f"[gen] loading model {args.pth} (L{cfg['num_hidden_layers']} "
          f"D{cfg['hidden_size']} N{cfg['head_dim']}, fp16)...")
    model = BatchedRWKV7(args.pth, cfg["num_hidden_layers"], cfg["hidden_size"],
                         cfg["head_dim"], device=args.device)
    gen = torch.Generator(device=args.device).manual_seed(args.seed)

    buffer_out = []
    n_seq = n_tok = 0
    t0 = time.time()
    for block in _prompt_stream(args, tokenizer):
        arr, bseq, btok = _generate_block(
            model, block, max_new=args.max_new, temperature=args.temperature,
            top_p=args.top_p, top_k=args.top_k, generator=gen)
        buffer_out.append(arr)
        n_seq += bseq
        n_tok += btok
        if n_seq % (args.batch * args.log_every) == 0:
            dt = time.time() - t0
            print(f"[gen] {n_seq} seqs, {n_tok:,} tokens, "
                  f"{n_tok / max(dt, 1e-6):.0f} tok/s, {dt / 60:.1f} min")
            if tracker is not None:
                tracker.log_now({"seqs": n_seq, "tokens": n_tok,
                                 "tok_s": n_tok / max(dt, 1e-6),
                                 "minutes": dt / 60.0})
    arr = np.concatenate(buffer_out) if buffer_out else np.zeros(0, dtype=np.uint16)
    arr.tofile(args.out)
    _write_meta(args, n_seq, int(arr.size), t0, extra={"devices": [args.device]})
    if tracker is not None:
        tracker.log_now({"done": 1, "seqs": n_seq, "tokens": int(arr.size),
                         "minutes": (time.time() - t0) / 60.0})
        tracker.finish()


def _write_meta(args, n_seq, n_tok, t0, extra):
    meta = {
        "source": "allenai/tulu-3-sft-mixture",
        "filtered_source": "ai2-adapt-dev/flan_v2_converted",
        "num_seqs": int(n_seq),
        "num_tokens": int(n_tok),
        "max_new": args.max_new,
        "max_prompt": args.max_prompt,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "dtype": "uint16",
        "model": os.path.abspath(args.pth),
        "seconds": round(time.time() - t0, 1),
    }
    meta.update(extra)
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[gen] wrote {args.out}: {meta['num_seqs']} seqs, {meta['num_tokens']:,} tokens "
          f"({n_tok * 2 / 1e6:.1f} MB) in {meta['seconds'] / 60:.1f} min")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve_devices(args):
    """--devices: 'auto' (one worker per visible CUDA GPU) | 'single' (just
    --device) | comma list like 'cuda:0,cuda:1'."""
    spec = (args.devices or "auto").strip().lower()
    if spec == "single":
        return [args.device]
    if spec in ("", "auto", "all"):
        n = torch.cuda.device_count()
        if n == 0:
            raise SystemExit("[gen] no CUDA devices visible")
        return [f"cuda:{i}" for i in range(n)]
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if part.isdigit():
            part = f"cuda:{part}"
        try:
            torch.device(part)
        except Exception:
            raise SystemExit(f"[gen] bad device in --devices: {part!r}")
        if part not in out:  # two workers on one GPU would just OOM
            out.append(part)
    if not out:
        raise SystemExit("[gen] --devices parsed to an empty device list")
    return out


def main():
    ap = argparse.ArgumentParser(description="Generate self-distilled MTP data")
    ap.add_argument("--pth", default="rwkv7-g1k-2.9b-20260930-ctx25600.pth")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--vocab", default="rwkv_vocab_v20230424.txt")
    ap.add_argument("--out", default="data/tulu_selfgen.bin")
    ap.add_argument("--dataset", choices=("tulu", "ultradata"), default="tulu",
                    help="source corpus: 'tulu' = allenai/tulu-3-sft-mixture "
                         "(the original); 'ultradata' = the four no_think "
                         "configs of openbmb/UltraData-SFT-2605, an equal share "
                         "of --num-samples from each")
    ap.add_argument("--num-samples", type=int, default=50000,
                    help="total prompts to sample across the whole run")
    ap.add_argument("--max-prompt", type=int, default=1024)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0",
                    help="device for --smoke and for '--devices single'")
    ap.add_argument("--engine", choices=("v3a", "legacy"), default="v3a",
                    help="v3a: faster3a_2607 engine (CUDA-graph decode, fused "
                         "kernels); legacy: original in-process BatchedRWKV7")
    ap.add_argument("--v3a-wkv", choices=("fp16", "fp32io16"), default="fp16",
                    help="v3a WKV kernel mode (fp32io16 is more accurate but "
                         "slower, esp. at large batch)")
    ap.add_argument("--v3a-deltalog", action="store_true",
                    help="v3a only: DeltaLog WKV path (+3-7 pct decode on "
                         "fp16); requires --batch in the engine's tuned table "
                         "(8/16/32/64/256/512 for this model); runs eagerly "
                         "(no CUDA graph)")
    ap.add_argument("--devices", default="auto",
                    help="'auto' (default): one worker per visible CUDA GPU, "
                         "fed from a shared block queue; 'single': use only "
                         "--device in-process; or comma list like "
                         "'cuda:0,cuda:1'")
    ap.add_argument("--log-every", type=int, default=1,
                    help="log every N batches (single-GPU) / blocks (multi-GPU)")
    ap.add_argument("--trackio-project", default="",
                    help="trackio project name (LOCAL logging only; view with "
                         "'trackio show --project <name>'); empty (default) "
                         "disables trackio logging entirely")
    ap.add_argument("--trackio-run", default="",
                    help="trackio run name (default: the --out file stem; "
                         "re-running with the same name resumes the same run)")
    ap.add_argument("--smoke", action="store_true",
                    help="generate from one fixed prompt and print it, then exit")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    tokenizer = RWKV_TOKENIZER(args.vocab)

    if args.smoke:
        prompt = "User: Give me three uses of a paperclip.\n\nAssistant:"
        ids = tokenizer.encode(prompt)
        print(f"[gen] loading model {args.pth} (L{cfg['num_hidden_layers']} "
              f"D{cfg['hidden_size']} N{cfg['head_dim']}, fp16)...")
        model = BatchedRWKV7(args.pth, cfg["num_hidden_layers"],
                             cfg["hidden_size"], cfg["head_dim"],
                             device=args.device)
        gen = torch.Generator(device=args.device).manual_seed(args.seed)
        logits, state = model.forward(
            torch.tensor(ids, device=args.device).unsqueeze(0), None)
        out = []
        lg = logits[:, -1, :].float()
        for _ in range(min(args.max_new, 128)):
            tok = sample_batch(lg, args.temperature, args.top_p, args.top_k, gen)
            out.append(int(tok.item()))
            lg, state = model.forward(tok.view(1, 1), state)
            lg = lg[:, -1, :].float()
        print(prompt + tokenizer.decode(out))
        return

    devices = _resolve_devices(args)
    # trackio (optional): initialized ONLY in the main process -- the spawned
    # GPU workers never touch it.  Data generation has no checkpoints, so
    # records are logged directly as progress happens (no buffering).
    tracker = None
    if args.trackio_project:
        tracker = TrackioRun(
            args.trackio_project,
            name=args.trackio_run or os.path.splitext(os.path.basename(args.out))[0],
            config=dict(vars(args)),
            tag="gen",
        )
    if len(devices) <= 1:
        _run_single(args, cfg, tokenizer, tracker)
    else:
        _run_multi(args, cfg, tokenizer, devices, tracker)


if __name__ == "__main__":
    main()