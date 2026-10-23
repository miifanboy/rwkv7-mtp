#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_mtp_to_gguf.py -- pack an MTP head checkpoint (``train_mtp.py`` output)
into a GGUF *draft model* that llama.cpp can load with::

    llama-cli -m rwkv7-2.9b.gguf -md mtp_heads.gguf --spec-type draft-mtp ...

WHAT THIS PRODUCES
------------------
A standalone, loadable GGUF whose architecture is ``rwkv7`` but which contains
**no trunk layers at all** -- only the MTP head(s) plus the two big tensors the
head contractually needs from the frozen backbone (``token_embd`` and the
``output`` lm-head).  This mirrors the "GLM4 MTP-only" draft files that
llama.cpp's ``deepseek2`` loader already supports, and it is what makes
``-md <path>`` work as a *separate* model from the target.

The metadata is set up so that llama.cpp's existing auto-detection succeeds:

    rwkv7.block_count           = K
    rwkv7.nextn_predict_layers  = K

so ``n_layer() == block_count - nextn_predict_layers == 0``: every block in the
file is a NextN block, indexed ``blk.0 .. blk.{K-1}``.  Two independent
mechanisms then line up on their own:

  * ``common_speculative_types_from_gguf()`` probes for
    ``blk.{block_count-1}.nextn.eh_proj.weight`` and infers ``draft-mtp``.
  * the MTP driver reads ``n_mtp_layers`` from ``nextn_predict_layers`` and, when
    it is > 1, sets ``chain_heads = true`` and walks the heads via
    ``llama_set_nextn_layer_offset``.

TENSOR MAP  (--mtp-arch mlp, checkpoint key -> GGUF name)
---------------------------------------------------------
    {k}.ln_h.weight      (C,)     -> blk.{k}.nextn.hnorm.weight
    {k}.ln_h.bias        (C,)     -> blk.{k}.nextn.hnorm.bias
    {k}.ln_tok.weight    (C,)     -> blk.{k}.nextn.enorm.weight
    {k}.ln_tok.bias      (C,)     -> blk.{k}.nextn.enorm.bias
    {k}.proj.weight      (C,2C)   -> blk.{k}.nextn.eh_proj.weight
    {k}.mlp.0.weight     (C,)     -> blk.{k}.nextn.mlp_norm.weight
    {k}.mlp.0.bias       (C,)     -> blk.{k}.nextn.mlp_norm.bias
    {k}.mlp.1.weight     (H,C)    -> blk.{k}.nextn.mlp_up.weight
    {k}.mlp.3.weight     (C,H)    -> blk.{k}.nextn.mlp_down.weight

    emb.weight           (V,C)    -> token_embd.weight
    head.weight          (V,C)    -> output.weight

TENSOR MAP  (--mtp-arch rwkv7)
------------------------------
The head is ONE full RWKV-7 block (mtp.RWKV7MTPModule: the same fusion as the
MLP head, then src.model.Block(args, layer_id=0) -- layer-0 semantics: the block
has its own pre-block norm ln0 and anchors its own value, v_first = v).  Its
tensors are stored under blk.0.nextn.* and loaded by llama.cpp into the layer's
regular time_mix_*/channel_mix_*/attn_norm* slots; the block's WKV/token-shift
state lives in the draft context's recurrent memory.

    0.ln_h.* / 0.ln_tok.* / 0.proj.weight  -> hnorm/enorm/eh_proj (as above)
    0.block.ln0.{w,b}       (C,)     -> blk.0.nextn.block_ln0.{weight,bias}
    0.block.ln1.{w,b}       (C,)     -> blk.0.nextn.attn_norm.{weight,bias}
    0.block.ln2.{w,b}       (C,)     -> blk.0.nextn.attn_norm_2.{weight,bias}
    0.block.att.x_{r,w,k,v,a,g}      -> blk.0.nextn.time_mix_lerp_fused.weight
                                        (packed to (6,1,1,C), r,w,k,v,a,g order)
    0.block.att.w0          (1,1,C)  -> blk.0.nextn.time_mix_w0.weight  (f32, flat)
    0.block.att.w1/w2       (C,R)    -> blk.0.nextn.time_mix_w1/w2.weight (f16, transposed)
    0.block.att.a0          (1,1,C)  -> blk.0.nextn.time_mix_a0.weight  (f32, flat)
    0.block.att.a1/a2       (C,R)    -> blk.0.nextn.time_mix_a1/a2.weight (f16, transposed)
    0.block.att.v0/v1/v2             -> blk.0.nextn.time_mix_v0/v1/v2.weight
                                        (copies of a0/a1/a2 -- dead layer-0 slots
                                        aliased in the checkpoint; emitted so the
                                        file matches the trunk layout)
    0.block.att.g1/g2       (C,R)    -> blk.0.nextn.time_mix_g1/g2.weight (f16, transposed)
    0.block.att.k_k/k_a     (1,1,C)  -> blk.0.nextn.time_mix_k_k/k_a.weight (f32, flat)
    0.block.att.r_k         (H,N)    -> blk.0.nextn.time_mix_r_k.weight (f32, flat)
    0.block.att.{receptance,key,value,output}.weight (C,C) -> time_mix_*.weight (f16)
    0.block.att.ln_x.{w,b}  (C,)     -> blk.0.nextn.time_mix_ln{,.bias} (GroupNorm)
    0.block.ffn.x_k         (1,1,C)  -> blk.0.nextn.channel_mix_lerp_k.weight (f32, flat)
    0.block.ffn.key.weight  (4C,C)   -> blk.0.nextn.channel_mix_key.weight   (f16)
    0.block.ffn.value.weight (C,4C)  -> blk.0.nextn.channel_mix_value.weight (f16)

The GGUF's lora-rank / head-size metadata describe the HEAD BLOCK (not the
frozen backbone): the head block's low-rank dims come from the training-time
heuristic formulas (e.g. decay 128 for C=2560), not from config.json's 96/96/64.

STATEFUL HEAD: because the block carries a recurrent state, exactly ONE nextn
block is exported (blk.0) even for a shared-head checkpoint trained with a
recursion depth K > 1.  The trained recursion depth maps to
``--spec-draft-n-max`` at inference (the driver's default of 3 matches a
num_mtp=3 checkpoint).  chain_heads (one distinct block per draft step) is not
supported for stateful heads.

SHARED HEADS (--shared-head, FastMTP style)
-------------------------------------------
A shared-head checkpoint contains ONE MTP head (keys ``0.*``) that the training
code applies ``num_mtp`` times recursively.  llama.cpp's draft driver, however,
walks one *distinct* NextN block per draft step via
``llama_set_nextn_layer_offset`` (HANDOFF_RWKV7_MTP_LLAMACPP.md §R5), so a
loadable file must contain ``blk.0 .. blk.{K-1}``.

With ``--shared-head`` the converter therefore replicates the single head's
tensors into all K blocks verbatim and keeps ``block_count = nextn_predict_layers
= K``.  Identical weights per block *is* the shared-head semantics, so this
loads and runs correctly on the existing C++ with zero changes; the only cost
is ~K× file size for the (small) head tensors.

NO transposes: gguf-py stores the numpy array shape reversed into ggml's
ne-order, so a PyTorch (out, in) Linear is written as ne0=in, ne1=out -- exactly
what `ggml_mul_mat` wants.  (So `eh_proj` lands in the file as ne0=2C, ne1=C,
which is the reverse of what the C++ `create_tensor(..., {2*n_embd, n_embd})`
call would suggest; trust the file, not the intuition.)

The two backbone tensors are duplicated on purpose: the draft graph needs the
*same* embedding matrix and the *same* frozen lm-head the backbone uses (the MTP
heads were trained against the frozen head), and there is no cross-model
weight-sharing path in llama.cpp.

NOTES / LIMITS
--------------
* The MTP head applies the frozen lm-head to its hidden state **directly**, with
  no extra normalisation, so no ``output_norm`` tensor is emitted and the draft
  graph must NOT insert a norm before the head.
* ``ln_h``/``ln_tok``/``mlp.0``/``ln0``/``ln1``/``ln2``/``ln_x`` are full
  ``nn.LayerNorm`` (or GroupNorm for ln_x) -- the C++ graph must use
  ``LLM_NORM`` (not ``LLM_NORM_RMS``) and the trunk's group-norm path for ln_x.
* The concat order is ``[ln_h(h), ln_tok(tok)]`` (hidden first), which is the
  opposite of DeepSeek/GLM4's ``[enorm, hnorm]``.  Do not "fix" it; ``proj`` was
  trained with hidden in the first half.
* With ``--skip-backbone`` the file will NOT load (no vocab, no token_embd), so
  it is only useful for inspecting the head tensors.
* ``--mtp-arch rwkv7`` checkpoints must be shared-head (or num_mtp == 1); the
  export carries exactly one nextn block.

Usage
-----
    python scripts/convert_mtp_to_gguf.py \\
        --mtp  mtp_out/mtp_heads.pt \\
        --pth  rwkv7-g1k-2.9b-20260930-ctx25600.pth \\
        --vocab rwkv_vocab_v20230424.txt \\
        --out  mtp_out/mtp_heads.gguf
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
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import gguf
except ImportError:  # pragma: no cover
    sys.exit("the 'gguf' python package is required (pip install gguf)")


ARCH = "rwkv7"

# --- MTP wiring -------------------------------------------------------------
# These four are read unconditionally by llama_model_rwkv7::load_arch_hparams()
# (they are only *used* by the trunk), so they must be present even though the
# draft has no trunk.  Values are copied from config.json / the G1 checkpoints.
LORA_DECAY_RANK = 96
LORA_ICLR_RANK = 96
LORA_VALUE_RESIDUAL_MIX_RANK = 64
LORA_GATE_RANK = 320
WKV_HEAD_SIZE = 64
LAYER_NORM_EPS = 1e-5
CONTEXT_LENGTH = 1048576  # RWKV is not context limited; matches conversion/rwkv.py

# --- the frozen-backbone tensors the head contractually depends on ----------
BACKBONE_TENSORS = {
    "emb.weight":  "token_embd.weight",
    "head.weight": "output.weight",
}

# checkpoint key -> suffix under blk.{k}.nextn.
# value = (gguf_suffix, keep_fp32?)
HEAD_TENSOR_MAP: Dict[str, Tuple[str, bool]] = {
    "ln_h.weight":   ("hnorm.weight",    True),
    "ln_h.bias":     ("hnorm.bias",      True),
    "ln_tok.weight": ("enorm.weight",    True),
    "ln_tok.bias":   ("enorm.bias",      True),
    "proj.weight":   ("eh_proj.weight",  False),
    "mlp.0.weight":  ("mlp_norm.weight", True),
    "mlp.0.bias":    ("mlp_norm.bias",   True),
    "mlp.1.weight":  ("mlp_up.weight",   False),
    "mlp.3.weight":  ("mlp_down.weight", False),
}

# --- --mtp-arch rwkv7 ---------------------------------------------------------
# The head is one full RWKV-7 block (mtp.RWKV7MTPModule). Its tensors live under
# blk.{k}.nextn.* and are loaded by llama.cpp into the layer's regular
# time_mix_*/channel_mix_*/attn_norm* slots so the draft graph can reuse the
# trunk build helpers verbatim. The block state (WKV + token shift) lives in the
# draft context's recurrent memory.
#
# mapping entry: (ckpt_suffix, gguf_suffix, kind) with kind:
#   "f32"    float32, stored as-is
#   "f16"    float16, stored as-is
#   "f16_t"  float16, transposed (PyTorch (out,in) Linear-style low-rank weights;
#            mirrors conversion/rwkv.py's time_mix_{w,a,v,g}{1,2} handling)
#   "f32_s"  float32, squeezed (1,1,C) -> (C,)
#   "f32_f"  float32, flattened (H,N) -> (H*N,)
RWKV7_HEAD_TENSOR_MAP: Tuple[Tuple[str, str, str], ...] = (
    # fusion (shared with the MLP head)
    ("ln_h.weight",                "hnorm.weight",                "f32"),
    ("ln_h.bias",                  "hnorm.bias",                  "f32"),
    ("ln_tok.weight",              "enorm.weight",                "f32"),
    ("ln_tok.bias",                "enorm.bias",                  "f32"),
    ("proj.weight",                "eh_proj.weight",              "f16"),
    # the block's own pre-block norm (Block.ln0, layer-0 semantics; the trunk
    # equivalent is the global token_embd_norm)
    ("block.ln0.weight",           "block_ln0.weight",            "f32"),
    ("block.ln0.bias",             "block_ln0.bias",              "f32"),
    # block norms: ln1 -> attn_norm (time-mix input), ln2 -> attn_norm_2 (ffn input)
    ("block.ln1.weight",           "attn_norm.weight",            "f32"),
    ("block.ln1.bias",             "attn_norm.bias",              "f32"),
    ("block.ln2.weight",           "attn_norm_2.weight",          "f32"),
    ("block.ln2.bias",             "attn_norm_2.bias",            "f32"),
    # token-shift lerp weights, packed r,w,k,v,a,g like the trunk converter
    ("block.att.x_r",              "time_mix_lerp_fused.weight",  "f32_pack_lerp"),
    # decay (w) / iclr (a) / gate (g) pathways
    ("block.att.w0",               "time_mix_w0.weight",          "f32_s"),
    ("block.att.w1",               "time_mix_w1.weight",          "f16_t"),
    ("block.att.w2",               "time_mix_w2.weight",          "f16_t"),
    ("block.att.a0",               "time_mix_a0.weight",          "f32_s"),
    ("block.att.a1",               "time_mix_a1.weight",          "f16_t"),
    ("block.att.a2",               "time_mix_a2.weight",          "f16_t"),
    ("block.att.g1",               "time_mix_g1.weight",          "f16_t"),
    ("block.att.g2",               "time_mix_g2.weight",          "f16_t"),
    # value-residual pathway: the head block is a layer-0 block (v_first = its
    # own value), so v0/v1/v2 are dead optimizer slots aliased onto a0/a1/a2 in
    # the checkpoint (mtp.RWKV7MTPModule.__init__). Emit them as copies of the
    # a-pathway tensors, exactly like conversion/rwkv.py does for backbone
    # layer 0 ("dummy v0/v1/v2 on first layer - easiest way to make llama happy").
    ("block.att.v0",               "time_mix_v0.weight",          "f32_s"),
    ("block.att.v1",               "time_mix_v1.weight",          "f16_t"),
    ("block.att.v2",               "time_mix_v2.weight",          "f16_t"),
    # k-side corrections
    ("block.att.k_k",              "time_mix_k_k.weight",         "f32_s"),
    ("block.att.k_a",              "time_mix_k_a.weight",         "f32_s"),
    ("block.att.r_k",              "time_mix_r_k.weight",         "f32_f"),
    # projections
    ("block.att.receptance.weight", "time_mix_receptance.weight", "f16"),
    ("block.att.key.weight",       "time_mix_key.weight",         "f16"),
    ("block.att.value.weight",     "time_mix_value.weight",       "f16"),
    ("block.att.output.weight",    "time_mix_output.weight",      "f16"),
    # group norm over heads (RWKV_Tmix_x070.ln_x)
    ("block.att.ln_x.weight",      "time_mix_ln.weight",          "f32"),
    ("block.att.ln_x.bias",        "time_mix_ln.bias",            "f32"),
    # channel mix
    ("block.ffn.x_k",              "channel_mix_lerp_k.weight",   "f32_s"),
    ("block.ffn.key.weight",       "channel_mix_key.weight",      "f16"),
    ("block.ffn.value.weight",     "channel_mix_value.weight",    "f16"),
)

LERP_ORDER = ("r", "w", "k", "v", "a", "g")

# the auto-detection marker checked by common_speculative_types_from_gguf()
DETECT_TENSOR = "blk.{}.nextn.eh_proj.weight"


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_mtp_payload(path: str) -> Tuple[Dict[str, torch.Tensor], int, int, dict, bool, str]:
    """Read ``mtp_heads.pt`` written by ``mtp.save_mtp_heads``.

    Returns ``(sd, num_mtp, n_embd, meta, shared_head, arch)``.  For a shared-head
    (FastMTP style) checkpoint ``num_mtp`` is the recursion depth K and the
    state dict holds a single head under ``0.*``.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "mtp_heads" not in payload:
        raise SystemExit(
            f"{path} does not look like an mtp_heads.pt "
            f"(top-level keys: {list(payload)[:8] if isinstance(payload, dict) else type(payload)})"
        )
    sd = {k: v for k, v in payload["mtp_heads"].items()}
    num_mtp = int(payload.get("num_mtp", 0))
    n_embd = int(payload.get("n_embd", 0))
    meta = payload.get("meta", {}) or {}
    shared_head = bool(payload.get("shared_head", False))
    arch = str(payload.get("arch") or meta.get("arch") or "mlp")
    if arch not in ("mlp", "rwkv7"):
        raise SystemExit(
            f"{path} declares --mtp-arch {arch!r}; supported export archs are "
            f"'mlp' and 'rwkv7'")
    if n_embd <= 0:
        raise SystemExit(f"{path} has no usable n_embd (got {n_embd})")
    return sd, num_mtp, n_embd, meta, shared_head, arch


def load_backbone_tensors(pth_path: str, keys: List[str]) -> Dict[str, torch.Tensor]:
    """Memory-map a backbone ``.pth`` and pull out just ``keys``.

    ``mmap=True`` keeps the 5.9 GB checkpoint on disk; only the two tensors we
    actually touch get paged in.
    """
    ckpt = torch.load(pth_path, map_location="cpu", weights_only=True, mmap=True)

    # unwrap the usual Lightning / HF wrappers
    if isinstance(ckpt, dict) and not all(
        isinstance(v, torch.Tensor) for v in list(ckpt.values())[:8]
    ):
        for wk in ("model", "module", "state_dict", "model_state_dict"):
            sub = ckpt.get(wk)
            if isinstance(sub, dict):
                ckpt = sub
                break

    out: Dict[str, torch.Tensor] = {}
    for prefix in ("", "_forward_module.", "module.", "rwkv7.", "rwkv."):
        for k in keys:
            if k in out:
                continue
            t = ckpt.get(prefix + k)
            if isinstance(t, torch.Tensor):
                out[k] = t
    missing = [k for k in keys if k not in out]
    if missing:
        raise SystemExit(
            f"{pth_path} is missing {missing}; available keys look like "
            f"{list(ckpt)[:8]}"
        )
    return out


def find_vocab(explicit: Optional[str]) -> str:
    if explicit:
        if not os.path.isfile(explicit):
            raise SystemExit(f"--vocab {explicit} does not exist")
        return explicit
    here = _ROOT  # project root (bootstrap above), not scripts/
    guesses = [
        os.path.join(here, "rwkv_vocab_v20230424.txt"),
        os.path.join(here, "..", "RWKV-v5", "tokenizer", "rwkv_vocab_v20230424.txt"),
        os.path.join(here, "..", "..", "RWKV-v5", "tokenizer", "rwkv_vocab_v20230424.txt"),
    ]
    for g in guesses:
        if os.path.isfile(g):
            return os.path.normpath(g)
    raise SystemExit(
        "could not find rwkv_vocab_v20230424.txt -- pass --vocab explicitly.\n"
        "  (it ships next to the RWKV HF model dir; this repo also has a copy at\n"
        "   ../../RWKV-v5/tokenizer/rwkv_vocab_v20230424.txt)"
    )


def read_rwkv_world_vocab(path: str, vocab_size: int) -> Tuple[List[bytes], List[int]]:
    """Re-implementation of ``_set_vocab_rwkv_world`` (conversion/base.py).

    Kept byte-compatible with llama.cpp's own converter so the draft GGUF's
    vocab is identical to the target's.
    """
    tokens: List[bytes] = ["<s>".encode("utf-8")]
    toktypes: List[int] = [gguf.TokenType.CONTROL]

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split(" ")
            if len(parts) < 3:
                continue
            token, token_len = ast.literal_eval(" ".join(parts[1:-1])), int(parts[-1])
            token = token.encode("utf-8") if isinstance(token, str) else token
            if len(token) != token_len:
                raise SystemExit(f"bad vocab line (length mismatch): {line!r}")
            # "b'\xff'" -> "\xff"
            tokens.append(repr(token)[2:-1].encode("utf-8"))
            toktypes.append(gguf.TokenType.NORMAL)

    if len(tokens) > vocab_size:
        raise SystemExit(f"vocab file has {len(tokens)} tokens but vocab_size={vocab_size}")
    for i in range(len(tokens), vocab_size):
        tokens.append(f"[PAD{i}]".encode("utf-8"))
        toktypes.append(gguf.TokenType.UNUSED)
    return tokens, toktypes


# ---------------------------------------------------------------------------
# conversion helpers
# ---------------------------------------------------------------------------

def to_numpy(t: torch.Tensor, keep_fp32: bool) -> np.ndarray:
    """torch tensor -> numpy, in PyTorch shape order (gguf-py reverses to ne-order).

    F16 for the big 2-D weights (bf16 -> f16 is a strict precision win: 10
    mantissa bits vs 7, and these weights are far from the f16 range limits),
    F32 for the 1-D norms where the values are O(1) and cheap to keep exact.
    """
    return to_numpy_kind(t, "f32" if keep_fp32 else "f16")


def to_numpy_kind(t: torch.Tensor, kind: str) -> np.ndarray:
    """torch tensor -> numpy with a per-tensor transform/precision spec.

    See RWKV7_HEAD_TENSOR_MAP for the kind vocabulary.  gguf-py stores the numpy
    shape reversed into ggml ne-order, so a PyTorch (out, in) Linear is written
    as ne0=in -- exactly what ggml_mul_mat wants.
    """
    t = t.detach().to("cpu")
    if kind == "f32":
        return t.float().contiguous().numpy()
    if kind == "f16":
        return t.to(torch.float16).contiguous().numpy()
    if kind == "f16_t":
        # low-rank time-mix weights: PyTorch (C, rank) -> ggml ne (C, rank)
        return t.to(torch.float16).t().contiguous().numpy()
    if kind in ("f32_s", "f32_f"):
        # (1,1,C) vectors and (H,N) correction matrices -> flat (C,)/(H*N,)
        return t.float().reshape(-1).contiguous().numpy()
    if kind == "f32_pack_lerp":
        return t.float().contiguous().numpy()
    raise SystemExit(f"unknown tensor kind {kind!r}")


def detect_num_mtp(sd: Dict[str, torch.Tensor], declared: int) -> int:
    idx = set()
    for k in sd:
        head, _, _ = k.partition(".")
        if head.isdigit():
            idx.add(int(head))
    if not idx:
        raise SystemExit("no '{index}.' prefixed keys found in mtp_heads")
    k = max(idx) + 1
    if declared and declared != k:
        print(f"[convert] note: checkpoint declares num_mtp={declared} but "
              f"contains heads 0..{k - 1}; using {k}")
    return k


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert an MTP head .pt into a llama.cpp draft GGUF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--mtp", required=True, help="mtp_out/mtp_heads.pt")
    ap.add_argument("--pth", help="frozen backbone .pth (provides token_embd + output)")
    ap.add_argument("--vocab", help="rwkv_vocab_v20230424.txt (auto-detected if omitted)")
    ap.add_argument("--out", required=True, help="output .gguf path")
    ap.add_argument("--skip-backbone",
                    action="store_true",
                    help="do NOT embed token_embd/output (result is not loadable)")
    ap.add_argument("--shared-head",
                    action="store_true",
                    help="checkpoint holds ONE MTP head (FastMTP style) applied "
                         "num_mtp times recursively; its tensors are replicated "
                         "into blk.0..blk.{K-1}.nextn so the existing C++ "
                         "chain_heads driver loads the file unchanged")
    ap.add_argument("--name", default="rwkv7-mtp", help="general.name")
    ap.add_argument("--quantize", default="none", choices=["none", "q8_0"],
                    help="quantize LARGE f16 tensors (token_embd, output, and the "
                         "head block's big 2-D weights) to Q8_0.  Halves the "
                         "per-decode memory traffic of the draft's lm-head GEMM "
                         "and shrinks the file ~40%%; norms / low-rank / small "
                         "tensors stay exact.  Draft-precision impact is ~0.2%% "
                         "relative (argmax flips only on near-ties).")
    ap.add_argument("--quantize-min-elements", type=int, default=1_000_000,
                    help="only tensors with at least this many elements are "
                         "quantized (needs --quantize q8_0)")
    args = ap.parse_args()

    sd, declared_num_mtp, n_embd, meta, ck_shared, arch = load_mtp_payload(args.mtp)
    shared_head = args.shared_head or ck_shared
    if args.shared_head and ck_shared:
        print(f"[convert] checkpoint declares shared_head=True (--shared-head redundant)")
    print(f"[convert] arch: {arch}")

    if arch == "rwkv7":
        # The rwkv7 head is STATEFUL (one RWKV-7 block with a recurrent WKV state).
        # llama.cpp's draft-mtp driver walks one distinct NextN block per draft step
        # (chain_heads) only for stateless heads; for a stateful head that path
        # cannot roll back the per-depth recurrent states, so the GGUF carries
        # exactly ONE NextN block and the trained recursion depth (shared-head
        # num_mtp) maps to --spec-draft-n-max at inference instead.
        if not shared_head and detect_num_mtp(sd, declared_num_mtp) > 1:
            raise SystemExit(
                "rwkv7-arch export supports only a single head: a shared-head "
                "checkpoint (FastMTP style) or num_mtp == 1. Re-train with "
                "--shared-head, or re-run with --mtp-arch mlp")
        if not any(k.startswith("0.") for k in sd):
            raise SystemExit("rwkv7 checkpoint has no '0.*' head tensors")
        num_mtp = 1
        if declared_num_mtp > 1:
            print(f"[convert] rwkv7 head: recursion depth {declared_num_mtp} -> "
                  f"run llama.cpp with --spec-draft-n-max {declared_num_mtp}")
    elif shared_head:
        # FastMTP style: one head, K recursive steps.  The depth comes from the
        # payload's num_mtp (there is only ever head '0' in the file).
        num_mtp = declared_num_mtp
        if num_mtp < 1:
            raise SystemExit(
                f"shared-head checkpoint declares num_mtp={num_mtp}; need >= 1"
            )
        if not any(k.startswith("0.") for k in sd):
            raise SystemExit("shared-head checkpoint has no '0.*' head tensors")
        print(f"[convert] shared-head mode: replicating ONE head into "
              f"blk.0..blk.{num_mtp - 1}.nextn (K={num_mtp})")
    else:
        num_mtp = detect_num_mtp(sd, declared_num_mtp)
    print(f"[convert] {args.mtp}: arch={arch} num_mtp={num_mtp} n_embd={n_embd} "
          f"step={meta.get('step')} tokens_seen={meta.get('tokens_seen')}")

    # ---- validate + collect head tensors ---------------------------------
    plan: List[Tuple[str, object, object]] = []  # (gguf name, tensor, kind/keep_fp32)
    if arch == "rwkv7":
        for ck_suffix, gg_suffix, kind in RWKV7_HEAD_TENSOR_MAP:
            if kind == "f32_pack_lerp":
                # pack x_r..x_g into one (6, 1, 1, C) tensor in r,w,k,v,a,g order
                # (same layout conversion/rwkv.py produces for the trunk; gguf-py
                # reverses it to ggml ne (C, 1, 1, 6))
                parts = []
                for c in LERP_ORDER:
                    t = sd.get(f"0.block.att.x_{c}")
                    if t is None:
                        raise SystemExit(f"missing tensor '0.block.att.x_{c}' in {args.mtp}")
                    parts.append(t.reshape(1, 1, 1, -1))
                packed = torch.cat(parts, dim=0).contiguous()   # (6, 1, 1, C)
                if packed.shape != (len(LERP_ORDER), 1, 1, n_embd):
                    raise SystemExit(
                        f"lerp packing produced {tuple(packed.shape)}, expected "
                        f"({len(LERP_ORDER)}, 1, 1, {n_embd})")
                plan.append((f"blk.0.nextn.time_mix_lerp_fused.weight", packed, "f32"))
                continue
            key = f"0.{ck_suffix}"
            t = sd.get(key)
            if t is None:
                if ck_suffix.startswith("block.att.v"):
                    # dead layer-0 slots; fall back to the a-pathway values they
                    # are aliased onto (mirrors conversion/rwkv.py layer 0)
                    t = sd.get("0." + ck_suffix.replace("att.v", "att.a"))
                if t is None:
                    raise SystemExit(f"missing tensor {key!r} in {args.mtp}")
            plan.append((f"blk.0.nextn.{gg_suffix}", t, kind))
    else:
        for k in range(num_mtp):
            src = 0 if shared_head else k   # shared: every block reads head '0'
            for ck_suffix, (gg_suffix, keep_fp32) in HEAD_TENSOR_MAP.items():
                key = f"{src}.{ck_suffix}"
                t = sd.get(key)
                if t is None:
                    # an MLP-free head is legal in the training code
                    if ck_suffix.startswith("mlp."):
                        continue
                    raise SystemExit(f"missing tensor {key!r} in {args.mtp}")
                plan.append((f"blk.{k}.nextn.{gg_suffix}", t,
                             "f32" if keep_fp32 else "f16"))

    # sanity: the head contract
    probe = sd.get("0.proj.weight")
    if probe is None or probe.ndim != 2:
        raise SystemExit("0.proj.weight missing or not 2-D")
    if probe.shape != (n_embd, 2 * n_embd):
        raise SystemExit(
            f"0.proj.weight has shape {tuple(probe.shape)}, expected "
            f"({n_embd}, {2 * n_embd}) -- n_embd={n_embd} looks wrong"
        )

    # per-arch hparams: for the rwkv7 head the GGUF's lora ranks / head size must
    # describe the HEAD BLOCK (there is no trunk in a draft file), and the head
    # block's low-rank dims come from the training-time heuristic formulas, not
    # from the backbone's config.json (e.g. decay 128 vs backbone 96).
    wkv_head_size = WKV_HEAD_SIZE
    lora_decay = LORA_DECAY_RANK
    lora_iclr = LORA_ICLR_RANK
    lora_value_res_mix = LORA_VALUE_RESIDUAL_MIX_RANK
    lora_gate = LORA_GATE_RANK
    ffn_size = 4 * n_embd
    if arch == "rwkv7":
        w1 = sd.get("0.block.att.w1")
        a1 = sd.get("0.block.att.a1")
        g1 = sd.get("0.block.att.g1")
        r_k = sd.get("0.block.att.r_k")
        ffn_key = sd.get("0.block.ffn.key.weight")
        if w1 is None or a1 is None or r_k is None:
            raise SystemExit("rwkv7 checkpoint missing block.att.w1/a1/r_k")
        lora_decay = int(w1.shape[1])
        lora_iclr = int(a1.shape[1])
        # layer-0 head: v0/v1/v2 are aliases of a0/a1/a2 (never used -- the block
        # anchors its own value); mirror the a-pathway rank like the loader does
        # for backbone layer 0
        lora_value_res_mix = lora_iclr
        lora_gate = int(g1.shape[1]) if g1 is not None else 0
        # r_k is (n_head, head_size)
        if r_k.ndim != 2 or r_k.shape[0] * r_k.shape[1] != n_embd:
            raise SystemExit(
                f"0.block.att.r_k has shape {tuple(r_k.shape)}; expected "
                f"(n_head, head_size) with n_head * head_size == n_embd={n_embd}")
        wkv_head_size = int(r_k.shape[1])
        if ffn_key is not None:
            ffn_size = int(ffn_key.shape[0])
        print(f"[convert] rwkv7 head block: decay_lora={lora_decay} iclr_lora={lora_iclr} "
              f"gate_lora={lora_gate} head_size={wkv_head_size} ffn={ffn_size}")
    else:
        has_mlp = any(k.endswith(".mlp.1.weight") for k in sd)
        if not has_mlp:
            print("[convert] WARNING: no MLP tensors -- the C++ graph must skip the MLP "
                  "branch for this file")

    # ---- vocab -----------------------------------------------------------
    vocab_tokens = vocab_types = None
    vocab_size = None
    if not args.skip_backbone:
        vocab_path = find_vocab(args.vocab)
        print(f"[convert] vocab: {vocab_path}")
        # the head/emb are (V, C); V comes from the checkpoint itself
        if not args.pth:
            raise SystemExit(
                "--pth is required (it supplies token_embd/output, and V from "
                "head.weight). Use --skip-backbone to omit them."
            )
        backbone = load_backbone_tensors(args.pth, list(BACKBONE_TENSORS))
        vocab_size = int(backbone["head.weight"].shape[0])
        for name, t in backbone.items():
            if t.shape != (vocab_size, n_embd):
                raise SystemExit(
                    f"{name} has shape {tuple(t.shape)}, expected ({vocab_size}, {n_embd})"
                )
        vocab_tokens, vocab_types = read_rwkv_world_vocab(vocab_path, vocab_size)

    # ---- write -----------------------------------------------------------
    writer = gguf.GGUFWriter(args.out, ARCH)
    writer.add_name(args.name)
    writer.add_context_length(CONTEXT_LENGTH)
    writer.add_embedding_length(n_embd)
    writer.add_block_count(num_mtp)            # every block is a NextN block
    writer.add_nextn_predict_layers(num_mtp)   # -> n_layer() = 0
    writer.add_layer_norm_eps(LAYER_NORM_EPS)
    writer.add_wkv_head_size(wkv_head_size)
    writer.add_decay_lora_rank(lora_decay)
    writer.add_iclr_lora_rank(lora_iclr)
    writer.add_value_residual_mix_lora_rank(lora_value_res_mix)
    if lora_gate > 0:
        writer.add_gate_lora_rank(lora_gate)
    writer.add_feed_forward_length(ffn_size)
    writer.add_head_count(0)
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16)

    if vocab_tokens is not None:
        writer.add_tokenizer_model("rwkv")
        writer.add_token_list(vocab_tokens)
        writer.add_token_types(vocab_types)
        writer.add_bos_token_id(0)
        writer.add_eos_token_id(0)
        writer.add_eot_token_id(261)
        writer.add_add_bos_token(False)

    def _finish(arr: np.ndarray, kind: str, what: str) -> np.ndarray:
        """Optional --quantize pass over one prepared numpy tensor."""
        if args.quantize != "q8_0":
            return arr
        if not kind.startswith("f16"):
            return arr                      # norms / lerp / low-rank stay exact
        if arr.nbytes // arr.itemsize < args.quantize_min_elements:
            return arr
        if arr.shape[-1] % 32 != 0:
            print(f"[convert] WARNING: {what} last dim {arr.shape[-1]} not "
                  f"Q8_0-aligned; keeping f16")
            return arr
        qarr = gguf.quants.quantize(arr.astype(np.float32),
                                    gguf.GGMLQuantizationType.Q8_0)
        return np.ascontiguousarray(qarr)

    nbytes = 0
    n_q = 0
    for item in plan:
        name, t, kind = item
        arr = to_numpy_kind(t, kind)
        raw = arr.nbytes
        arr = _finish(arr, kind, name)
        if arr.nbytes != raw:
            n_q += 1
            print(f"[convert] q8_0: {name} {tuple(arr.shape) if arr.ndim == 2 else ''} "
                  f"{raw / 2**20:.1f} -> {arr.nbytes / 2**20:.1f} MiB")
        raw_dtype = (gguf.GGMLQuantizationType.Q8_0
                     if arr.dtype == np.uint8 else None)
        writer.add_tensor(name, arr, raw_dtype=raw_dtype)
        nbytes += arr.nbytes

    if vocab_size is not None:
        for src, dst in BACKBONE_TENSORS.items():
            arr = to_numpy(backbone[src], keep_fp32=False)
            raw = arr.nbytes
            arr = _finish(arr, "f16", dst)
            raw_dtype = (gguf.GGMLQuantizationType.Q8_0
                         if arr.dtype == np.uint8 else None)
            writer.add_tensor(dst, arr, raw_dtype=raw_dtype)
            nbytes += arr.nbytes
            print(f"[convert] embedded {src} {tuple(backbone[src].shape)} -> {dst} "
                  f"({raw / 2**20:.1f} MiB"
                  + (f" -> q8_0 {arr.nbytes / 2**20:.1f} MiB" if arr.nbytes != raw else "")
                  + ")")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"[convert] wrote {args.out}: {len(plan)} head tensors ({n_q} quantized), "
          f"{nbytes / 2**20:.1f} MiB of tensor data"
          + (f" (shared head replicated {num_mtp}x)" if shared_head and arch == "mlp" else ""))
    if shared_head and arch == "mlp":
        print("[convert] NOTE: all K blocks hold identical weights by design "
              "(shared-head semantics; no C++ changes required)")
    if arch == "rwkv7":
        print("[convert] NOTE: rwkv7 head is stateful -- the draft context gets a "
              "recurrent memory; only ONE nextn block is exported (chain_heads is "
              "not supported for stateful heads). Recursion depth via --spec-draft-n-max.")
    print(f"[convert] auto-detect marker present: "
          f"{DETECT_TENSOR.format(num_mtp - 1)}")
    print()
    print(f"[convert] next: llama-cli -m <rwkv7-2.9b.gguf> -md {args.out} \\")
    print(f"        --spec-type draft-mtp -ngl 99 -c 4096 -n 256 -st --temp 0 --top-k 1"
          + (f" --spec-draft-n-max {declared_num_mtp}" if arch == "rwkv7" and declared_num_mtp > 1 else ""))


if __name__ == "__main__":
    main()
