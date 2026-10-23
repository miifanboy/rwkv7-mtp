#!/usr/bin/env python3
"""Generate tiny synthetic RWKV-7 GGUFs (a target and matching MTP drafts).

All files share the same vocab and n_embd so that llama.cpp's speculative
machinery accepts them (`common_speculative_are_compatible`). Weights are
random; the point is to exercise the *code paths* (draft-mtp driver + the
RWKV-7 MTP graph + the target t_h_nextn plumbing), not to produce sensible
text.

Two draft flavors are produced, matching the two --mtp-arch options of the
``mtp`` package:

    tiny/mtp.gguf        --mtp-arch mlp   (stateless fusion+MLP head)
    tiny/mtp_block.gguf  --mtp-arch rwkv7 (stateful fusion+RWKV-7 block head;
                                     one nextn block, recurrent state)

Usage:
    python scripts/make_tiny_rwkv7.py --out-dir tiny --embd 64 --layer 2

Produces:
    tiny/target.gguf
    tiny/mtp.gguf
    tiny/mtp_block.gguf
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
import os

import numpy as np

import gguf

# reuse the exact RWKV-world vocab handling of the MTP converter
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_mtp_to_gguf import read_rwkv_world_vocab, find_vocab, LAYER_NORM_EPS, CONTEXT_LENGTH


def rnd(shape, scale=0.02, rng=None):
    return (rng.standard_normal(shape).astype(np.float32) * scale)


def ones(shape):
    return np.ones(shape, dtype=np.float32)


def zeros(shape):
    return np.zeros(shape, dtype=np.float32)


class Writer:
    def __init__(self, path, n_embd, n_vocab, n_layer, block_count, nextn, tok_writer):
        self.w = gguf.GGUFWriter(path, "rwkv7")
        self.w.add_name(os.path.basename(path))
        self.w.add_context_length(CONTEXT_LENGTH)
        self.w.add_embedding_length(n_embd)
        self.w.add_block_count(block_count)
        if nextn:
            self.w.add_nextn_predict_layers(nextn)
        self.w.add_layer_norm_eps(LAYER_NORM_EPS)
        self.w.add_wkv_head_size(64)
        self.w.add_decay_lora_rank(8)
        self.w.add_iclr_lora_rank(8)
        self.w.add_value_residual_mix_lora_rank(8)
        self.w.add_gate_lora_rank(8)
        self.w.add_feed_forward_length(128)
        self.w.add_head_count(0)
        self.w.add_file_type(gguf.LlamaFileType.ALL_F32)
        tok_writer(self.w)
        self.n = 0

    def add(self, name, *ne):
        # ne is the ggml ne-order the C++ loader expects; gguf-py reverses the
        # numpy shape into ne-order, so build numpy in reverse.
        arr = rnd(tuple(reversed(ne)))
        self.w.add_tensor(name, arr)
        self.n += 1
        return arr

    def add_arr(self, name, arr):
        self.w.add_tensor(name, arr)
        self.n += 1

    def close(self):
        self.w.write_header_to_file()
        self.w.write_kv_data_to_file()
        self.w.write_tensors_to_file()
        self.w.close()


def add_vocab(w, V, vocab_path):
    tokens, types = read_rwkv_world_vocab(vocab_path, V)
    w.add_tokenizer_model("rwkv")
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_bos_token_id(0)
    w.add_eos_token_id(0)
    w.add_eot_token_id(261)
    w.add_add_bos_token(False)


def gen_target(path, C, L, V, vocab_path):
    def add_vocab_(w):
        add_vocab(w, V, vocab_path)

    w = Writer(path, C, V, L, L, 0, add_vocab_)
    rng = np.random.default_rng(11)

    w.add_arr("token_embd.weight",      rnd((V, C), 0.02, rng))
    w.add_arr("token_embd_norm.weight", ones((C,)))
    w.add_arr("token_embd_norm.bias",   zeros((C,)))
    w.add_arr("output_norm.weight",     ones((C,)))
    w.add_arr("output_norm.bias",       zeros((C,)))
    w.add_arr("output.weight",          rnd((V, C), 0.02, rng))

    ffn = 128
    for i in range(L):
        p = f"blk.{i}."
        w.add_arr(p + "attn_norm.weight",   ones((C,)))
        w.add_arr(p + "attn_norm.bias",     zeros((C,)))
        w.add_arr(p + "attn_norm_2.weight", ones((C,)))
        w.add_arr(p + "attn_norm_2.bias",   zeros((C,)))

        # time-mix lora/lerp
        w.add_arr(p + "time_mix_w0.weight", zeros((C,)))
        w.add_arr(p + "time_mix_w1.weight", rnd((8, C), 0.02, rng))
        w.add_arr(p + "time_mix_w2.weight", rnd((C, 8), 0.02, rng))
        w.add_arr(p + "time_mix_a0.weight", zeros((C,)))
        w.add_arr(p + "time_mix_a1.weight", rnd((8, C), 0.02, rng))
        w.add_arr(p + "time_mix_a2.weight", rnd((C, 8), 0.02, rng))
        w.add_arr(p + "time_mix_v0.weight", zeros((C,)))
        w.add_arr(p + "time_mix_v1.weight", rnd((8, C), 0.02, rng))
        w.add_arr(p + "time_mix_v2.weight", rnd((C, 8), 0.02, rng))
        w.add_arr(p + "time_mix_g1.weight", rnd((8, C), 0.02, rng))
        w.add_arr(p + "time_mix_g2.weight", rnd((C, 8), 0.02, rng))
        w.add_arr(p + "time_mix_lerp_fused.weight", zeros((6, 1, 1, C)))
        w.add_arr(p + "time_mix_k_k.weight", ones((C,)))
        w.add_arr(p + "time_mix_k_a.weight", rnd((C,), 0.02, rng))
        w.add_arr(p + "time_mix_r_k.weight", ones((C,)))
        w.add_arr(p + "time_mix_key.weight",        rnd((C, C), 0.02, rng))
        w.add_arr(p + "time_mix_value.weight",      rnd((C, C), 0.02, rng))
        w.add_arr(p + "time_mix_receptance.weight", rnd((C, C), 0.02, rng))
        w.add_arr(p + "time_mix_ln.weight",   ones((C,)))
        w.add_arr(p + "time_mix_ln.bias",     zeros((C,)))
        w.add_arr(p + "time_mix_output.weight", rnd((C, C), 0.02, rng))

        # channel mix
        w.add_arr(p + "channel_mix_lerp_k.weight", zeros((1, 1, C)))
        w.add_arr(p + "channel_mix_key.weight",   rnd((ffn, C), 0.02, rng))
        w.add_arr(p + "channel_mix_value.weight", rnd((C, ffn), 0.02, rng))

    w.close()
    print(f"[tiny] {path}: {w.n} tensors (target C={C} L={L} V={V})")


def gen_mtp(path, C, K, V, vocab_path):
    def add_vocab_(w):
        add_vocab(w, V, vocab_path)

    # every block is a NextN block -> n_layer() == 0
    w = Writer(path, C, V, 0, K, K, add_vocab_)
    rng = np.random.default_rng(22)

    w.add_arr("token_embd.weight", rnd((V, C), 0.02, rng))
    w.add_arr("output.weight",     rnd((V, C), 0.02, rng))

    for k in range(K):
        p = f"blk.{k}.nextn."
        w.add_arr(p + "hnorm.weight", ones((C,)))
        w.add_arr(p + "hnorm.bias",   zeros((C,)))
        w.add_arr(p + "enorm.weight", ones((C,)))
        w.add_arr(p + "enorm.bias",   zeros((C,)))
        w.add_arr(p + "eh_proj.weight", rnd((C, 2*C), 0.02, rng))
        w.add_arr(p + "mlp_norm.weight", ones((C,)))
        w.add_arr(p + "mlp_norm.bias",   zeros((C,)))
        w.add_arr(p + "mlp_up.weight",   rnd((2*C, C), 0.02, rng))
        w.add_arr(p + "mlp_down.weight", rnd((C, 2*C), 0.02, rng))

    w.close()
    print(f"[tiny] {path}: {w.n} tensors (mtp C={C} K={K} V={V})")


def gen_mtp_block(path, C, K, V, vocab_path):
    """--mtp-arch rwkv7 draft: fusion + one full RWKV-7 block under blk.0.nextn.*

    Exactly ONE nextn block is emitted (the stateful head is driven through the
    driver's single-head recursion; chain_heads cannot roll back recurrent
    state). Tensor names/shapes mirror convert_mtp_to_gguf.py's rwkv7 mapping
    and the llama.cpp rwkv7 loader's expectations:
      - time_mix_lerp_fused packs x_r..x_g as ggml ne (C, 1, 1, 6)
      - low-rank time-mix weights are stored transposed (ne (C, rank))
      - r_k is flattened to (C,)
      - v0/v1/v2 are emitted like conversion/rwkv.py layer 0 (copies of a*)
    """
    def add_vocab_(w):
        add_vocab(w, V, vocab_path)

    rank = 8
    heads = 1
    head_size = C // heads
    ffn = 128

    w = Writer(path, C, V, 0, 1, 1, add_vocab_)
    rng = np.random.default_rng(33)

    w.add_arr("token_embd.weight", rnd((V, C), 0.02, rng))
    w.add_arr("output.weight",     rnd((V, C), 0.02, rng))

    p = "blk.0.nextn."
    # fusion (shared with the mlp head)
    w.add_arr(p + "hnorm.weight", ones((C,)))
    w.add_arr(p + "hnorm.bias",   zeros((C,)))
    w.add_arr(p + "enorm.weight", ones((C,)))
    w.add_arr(p + "enorm.bias",   zeros((C,)))
    w.add_arr(p + "eh_proj.weight", rnd((C, 2*C), 0.02, rng))
    # block norms (ln0/ln1/ln2)
    w.add_arr(p + "block_ln0.weight",   ones((C,)))
    w.add_arr(p + "block_ln0.bias",     zeros((C,)))
    w.add_arr(p + "attn_norm.weight",   ones((C,)))
    w.add_arr(p + "attn_norm.bias",     zeros((C,)))
    w.add_arr(p + "attn_norm_2.weight", ones((C,)))
    w.add_arr(p + "attn_norm_2.bias",   zeros((C,)))
    # token-shift lerp weights packed r,w,k,v,a,g -> ne (C, 1, 1, 6)
    w.add_arr(p + "time_mix_lerp_fused.weight", zeros((6, 1, 1, C)))
    # decay (w) / iclr (a) / gate (g) pathways -- low-rank weights transposed
    w.add_arr(p + "time_mix_w0.weight", zeros((C,)))
    w.add_arr(p + "time_mix_w1.weight", rnd((rank, C), 0.02, rng))
    w.add_arr(p + "time_mix_w2.weight", rnd((C, rank), 0.02, rng))
    w.add_arr(p + "time_mix_a0.weight", zeros((C,)))
    w.add_arr(p + "time_mix_a1.weight", rnd((rank, C), 0.02, rng))
    w.add_arr(p + "time_mix_a2.weight", rnd((C, rank), 0.02, rng))
    w.add_arr(p + "time_mix_g1.weight", rnd((rank, C), 0.02, rng))
    w.add_arr(p + "time_mix_g2.weight", rnd((C, rank), 0.02, rng))
    # layer-0 value-residual dummies (copies of a*)
    w.add_arr(p + "time_mix_v0.weight", zeros((C,)))
    w.add_arr(p + "time_mix_v1.weight", rnd((rank, C), 0.02, rng))
    w.add_arr(p + "time_mix_v2.weight", rnd((C, rank), 0.02, rng))
    # k-side corrections
    w.add_arr(p + "time_mix_k_k.weight", ones((C,)))
    w.add_arr(p + "time_mix_k_a.weight", rnd((C,), 0.02, rng))
    w.add_arr(p + "time_mix_r_k.weight", ones((C,)))
    # projections
    w.add_arr(p + "time_mix_key.weight",        rnd((C, C), 0.02, rng))
    w.add_arr(p + "time_mix_value.weight",      rnd((C, C), 0.02, rng))
    w.add_arr(p + "time_mix_receptance.weight", rnd((C, C), 0.02, rng))
    w.add_arr(p + "time_mix_ln.weight",   ones((C,)))
    w.add_arr(p + "time_mix_ln.bias",     zeros((C,)))
    w.add_arr(p + "time_mix_output.weight", rnd((C, C), 0.02, rng))
    # channel mix
    w.add_arr(p + "channel_mix_lerp_k.weight", zeros((1, 1, C)))
    w.add_arr(p + "channel_mix_key.weight",   rnd((ffn, C), 0.02, rng))
    w.add_arr(p + "channel_mix_value.weight", rnd((C, ffn), 0.02, rng))

    w.close()
    print(f"[tiny] {path}: {w.n} tensors (mtp block C={C} heads={heads} "
          f"head_size={head_size} rank={rank} V={V})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="tiny")
    ap.add_argument("--embd", type=int, default=64)
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--mtp", type=int, default=1)
    ap.add_argument("--vocab", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    vocab_path = find_vocab(args.vocab)
    print(f"[tiny] vocab: {vocab_path}")

    V = 65536
    gen_target(os.path.join(args.out_dir, "target.gguf"), args.embd, args.layer, V, vocab_path)
    gen_mtp(os.path.join(args.out_dir, "mtp.gguf"), args.embd, args.mtp, V, vocab_path)
    gen_mtp_block(os.path.join(args.out_dir, "mtp_block.gguf"), args.embd, 1, V, vocab_path)


if __name__ == "__main__":
    main()
