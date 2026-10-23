"""
estimate_vram_data.py -- analytical VRAM estimate for scripts/generate_data.py.

Takes the SAME command line as generate_data.py, never builds a model, never
touches CUDA, and prints a per-component VRAM breakdown derived from closed
formulas over (config dims, --batch, --max-new, --max-prompt, engine flags).

The formulas mirror the actual code paths of both backends:

  * weights     : the full checkpoint (bf16 on disk) resident on GPU as fp16
                  (DTYPE = torch.float16 in both engines).
                  --engine v3a: the engine's LOWRANK_WEIGHT default is "both",
                  so every low-rank tensor (att.w1/w2/a1/a2/v1/v2/g1/g2) is
                  stored TWICE (original + .t() transpose): +2*C*(Dd+Da+Dv+Dg)
                  elements per layer.  EMB_DEVICE="gpu" keeps the ln0-fused
                  embedding on GPU (counted once).  RKV_MODE="off" and
                  CMIX_SPARSE="no-fc" add no extra copies.
                  --engine legacy: BatchedRWKV7 aliases layer-0 att.v0/v1/v2
                  onto a0/a1/a2, so layer-0 v1/v2 tensors are freed
                  (-2*C*Dv elements).
  * rnn state   : per layer, two (B,C) shift buffers (fp16) + one WKV state
                  (B,H,N,N): fp32 in the legacy engine, fp16 in v3a's default
                  --v3a-wkv fp16 mode (fp32io16 keeps fp32).  H*N = C, so the
                  WKV state is B*C*N elements -> this term DOMINATES at large
                  --batch (e.g. 1.5B model, B=768: ~4.6 GiB fp16 / 9.2 GiB
                  fp32 per GPU).  --batch is the hard VRAM knob; --grad-acc
                  does not exist here and --num-samples / --max-new do not
                  change peak VRAM (tok_hist grows with --max-new but is tiny).
  * logits      : v3a keeps persistent (B,V) fp32 logits_buf + (B,V) fp16
                  graph output + (max_new,B) int64 tok_hist; its sampler
                  top-k(512)s instead of full-vocab sorting (4*B*V fp32 copy
                  + ~24*B*k transients).  legacy runs softmax + full-vocab
                  top-p sort over (B,V) fp32 (~32*B*V transient at
                  --top-p in (0,1), less for pure top-k), plus the fp32
                  (B,V) step logits.
  * graph pool  : v3a captures the T=1 decode step into a CUDA graph; the
                  graph's private memory pool retains the step's transient
                  allocations (~32*B*C + 2*B*V bytes) for the process life.
                  --v3a-deltalog runs eagerly instead (no graph pool) but
                  adds the DeltaLog workspace: 5*(M-1)*B*C*2*L bytes, with
                  the tuned merge interval M from the engine's table.
  * prefill     : sequential B=1 per prompt (RWKV is a pure RNN): one
                  single-sequence state + ~(1,max_prompt,C) transients;
                  legacy additionally accumulates last_logits (B,V) fp32 and
                  torch.cat's them (2x (B,V) fp32 transient).

Peak = per-GPU worker peak.  Multi-GPU runs (--devices auto) spawn ONE worker
per CUDA GPU, each holding exactly this footprint on its own card; the total
across N GPUs is N x this estimate (plus each worker's CUDA context).
v3a's extra cuBLAS/cublasLt workspaces are added separately.

Example
-------
    python scripts/estimate_vram_data.py \
        --dataset ultradata --num-samples 300000 --max-new 3072 --batch 768 \
        --out data-ob-1.5b/train_300.bin --config config-1.5b.json \
        --pth rwkv7-g1k-1.5b-20260930-ctx25600.pth
    # legacy backend:
    python scripts/estimate_vram_data.py --engine legacy --batch 256 ...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

MIB = 2**20

# ---------------------------------------------------------------------------
# Tunable constants (documented, keep in sync with generate_data.py /
# faster3a_2607/rwkv7_fast_v3a.py)
# ---------------------------------------------------------------------------

HEAD_SIZE_DEFAULT = 64          # faster3a_2607/rwkv7_fast_v3a.py HEAD_SIZE
GEN_DTYPE_BYTES = 2             # torch.float16 for ALL GPU-resident data

# v3a engine defaults (rwkv7_fast_v3a.py module globals, not overridden by
# generate_data.py): every low-rank weight is kept as orig AND transpose.
V3A_LOW_RANK_SUFFIX_BYTES_PER_LAYER = 2 * 2   # elems x fp16 -> per C*Lr unit
V3A_CUBLAS_WORKSPACE_MIB = 64.0   # cublasLt/cuBLAS workspace, handles, kernels

# legacy sample_batch(): softmax -> top-p full-vocab sort.  Simultaneously
# live (B,V)-sized buffers at peak (fp32 logits kept by the caller count
# separately): probs + sorted vals (fp32) + sorted idx (int64) + cumsum +
# masked copy + scatter target ~ 28*B*V bytes, plus 2*B*V bool/drop temporaries.
# (units: BYTES per (B,V) element -- i.e. 30 ~ 7.5 simultaneously-live
# fp32 (B,V) buffers + int64 idx; NOT a multiplier on fp32())
LEGACY_SAMPLING_TOPP_BV_BYTES = 30.0
LEGACY_SAMPLING_TOPK_BV_BYTES = 12.0   # probs + zeroed probs + kth (no sort)
# legacy decode step transients: ~16 (B,1,C)-shaped fp16 mix/ffn tensors.
LEGACY_DECODE_BF16_UNITS = 16.0        # x B*C fp16
# v3a captured decode graph pool: the T=1 step's transient allocations are
# retained by the graph's private mempool for the process lifetime
# (same ~16 (B,C) fp16 units + the fp16 (B,V) logits buffer).
V3A_GRAPH_BF16_UNITS = 16.0
# legacy prefill: last_logits accumulates (B,V) fp32, torch.cat doubles it.
LEGACY_PREFILL_LAST_LOGITS_BV_FP32 = 2.0
# per-prompt forward transients: ~16 (1,T,C)-shaped fp16 tensors.
PREFILL_BF16_UNITS = 16.0
# v3a _v3a_sample(): fp32 logits copy + topk vals/ids/probs/cdf/out over
# k=min(top_k or 512, V).
V3A_SAMPLING_BV_BYTES = 4.0            # logits.float()
V3A_SAMPLING_TOPK_BK_BYTES = 24.0      # fp32 vals+probs+cdf + int64 ids+out

# nvidia-smi overhead beyond torch's caching-allocator reserved pool
# (CUDA context, kernel module images; cuBLAS workspaces counted separately).
CUDA_CONTEXT_MIB_DEFAULT = 150.0
# torch reserved > allocated (caching-allocator block rounding/fragmentation).
FRAGMENTATION_PCT_DEFAULT = 4.3

# DeltaLog tuned merge intervals, copied from
# faster3a_2607/rwkv7_fast_v3a.py WKV_DELTALOG_TUNED_M (keep in sync).
WKV_DELTALOG_TUNED_M = {
    (768, 16): 2, (768, 32): 3, (768, 64): 3, (768, 128): 3,
    (768, 256): 3, (768, 512): 3,
    (1024, 16): 2, (1024, 32): 3, (1024, 64): 3,
    (1024, 256): 3, (1024, 512): 3,
    (2048, 8): 2, (2048, 16): 3, (2048, 32): 3, (2048, 64): 3,
    (2048, 256): 3, (2048, 512): 4,
    (2560, 8): 2, (2560, 16): 3, (2560, 32): 3, (2560, 64): 3,
    (2560, 256): 3, (2560, 512): 4,
    (4096, 8): 2, (4096, 16): 3, (4096, 32): 3, (4096, 64): 3,
    (4096, 128): 3, (4096, 256): 3, (4096, 512): 4,
}


# ---------------------------------------------------------------------------
# args -- identical to generate_data.py (unknown args are ignored)
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Estimate generate_data.py VRAM usage (analytical, no GPU)")
    # affect VRAM or the printed report
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--pth", default="rwkv7-g1k-2.9b-20260930-ctx25600.pth")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--max-prompt", type=int, default=1024)
    ap.add_argument("--engine", default="v3a", choices=("v3a", "legacy"))
    ap.add_argument("--v3a-wkv", default="fp16", choices=("fp16", "fp32io16"))
    ap.add_argument("--v3a-deltalog", action="store_true")
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)
    # accepted for CLI compatibility, do not influence VRAM
    ap.add_argument("--vocab", default="rwkv_vocab_v20230424.txt")
    ap.add_argument("--out", default="data/tulu_selfgen.bin")
    ap.add_argument("--dataset", default="tulu")
    ap.add_argument("--num-samples", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--devices", default="auto")
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    # estimator knobs
    ap.add_argument("--cuda-overhead-mib", type=float,
                    default=CUDA_CONTEXT_MIB_DEFAULT,
                    help="nvidia-smi overhead beyond the torch allocator pool")
    ap.add_argument("--frag-pct", type=float,
                    default=FRAGMENTATION_PCT_DEFAULT,
                    help="torch reserved-over-allocated fragmentation, %%")
    ap.add_argument("--headroom-pct", type=float, default=0.0,
                    help="extra safety headroom added to the final total, %%")
    ap.add_argument("--verify-pth", action="store_true",
                    help="load the .pth on CPU and cross-check the parameter "
                         "count (slow-ish, exact)")
    args, _unknown = ap.parse_known_args(argv)
    return args


# ---------------------------------------------------------------------------
# dims + parameter formulas (same backbone formula as estimate_vram.py,
# verified bit-exact against the G1 checkpoints)
# ---------------------------------------------------------------------------

def load_dims(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    C = int(cfg.get("hidden_size", cfg.get("n_embd")))
    V = int(cfg["vocab_size"])
    L = int(cfg.get("num_hidden_layers", cfg.get("n_layer")))
    N = int(cfg.get("head_dim", cfg.get("head_size", HEAD_SIZE_DEFAULT)))
    F = int(cfg.get("intermediate_size", int(C * 3.5) // 32 * 32))
    Dd = int(cfg.get("decay_low_rank_dim", _lora(2.5, C)))
    Da = int(cfg.get("a_low_rank_dim", _lora(2.5, C)))
    Dv = int(cfg.get("v_low_rank_dim", _lora(1.7, C)))
    Dg = int(cfg.get("gate_low_rank_dim", _lora(5.0, C)))
    return C, V, L, N, F, Dd, Da, Dv, Dg


def _lora(mult: float, C: int) -> int:
    return max(32, int(round(mult * math.sqrt(C) / 32)) * 32)


def block_params(C, F, Dd, Da, Dv, Dg, ln0):
    """One RWKV-7 block (RWKV_Tmix_x070 + RWKV_CMix_x070 + norms)."""
    n = 4 * C + (2 * C if ln0 else 0)          # ln1, ln2 (+ ln0)
    n += 6 * C                                  # x_r x_w x_k x_v x_a x_g
    n += C + C * Dd + Dd * C                    # w0, w1, w2
    n += C + C * Da + Da * C                    # a0, a1, a2
    n += C + C * Dv + Dv * C                    # v0, v1, v2
    n += C * Dg + Dg * C                        # g1, g2
    n += C + C + C                              # k_k, k_a, r_k
    n += 4 * C * C + 2 * C                      # receptance/key/value/output + ln_x
    n += C + 2 * C * F                          # ffn x_k, key, value
    return n


def backbone_params(C, V, L, F, Dd, Da, Dv, Dg):
    n = V * C            # emb
    n += V * C           # head (tied=False in the G1 checkpoints)
    n += 2 * C           # ln_out
    for i in range(L):
        n += block_params(C, F, Dd, Da, Dv, Dg, ln0=(i == 0))
    return n


# ---------------------------------------------------------------------------
# VRAM estimate
# ---------------------------------------------------------------------------

def fmt(mib):
    if mib >= 1024:
        return f"{mib / 1024:.2f} GiB"
    return f"{mib:.1f} MiB"


def main(argv=None):
    a = parse_args(argv)
    C, V, L, N, F, Dd, Da, Dv, Dg = load_dims(a.config)
    B = max(1, a.batch)
    Tnew = max(1, a.max_new)
    Tprompt = max(1, a.max_prompt)
    v3a = a.engine == "v3a"
    b = GEN_DTYPE_BYTES

    # element helpers
    p = lambda n: n * b / MIB            # fp16/fp32 param-dtype elements -> MiB
    fp32 = lambda n: n * 4 / MIB
    int64 = lambda n: n * 8 / MIB

    # -- weights ----------------------------------------------------------
    total_params = backbone_params(C, V, L, F, Dd, Da, Dv, Dg)
    weights_mib = p(total_params)
    if v3a:
        # LOWRANK_WEIGHT="both": orig + transpose for w1/w2/a1/a2/v1/v2/g1/g2
        lowrank_per_layer = 2 * C * (Dd + Da + Dv + Dg)
        v3a_lowrank_mib = p(lowrank_per_layer * L)
    else:
        # layer-0 att.v1/v2 aliased onto a1/a2 and freed
        lowrank_per_layer = 0
        v3a_lowrank_mib = -p(2 * C * Dv)

    # -- RNN state (the dominant --batch term) -----------------------------
    # per layer: 2x (B,C) fp16 shift + (B,H,N,N) WKV state (H*N = C)
    wkv_state_bytes = (2 if (v3a and a.v3a_wkv == "fp16") else 4)
    shift_mib = p(L * 2 * B * C)
    wkv_mib = fp32(L * B * C * N) * (wkv_state_bytes / 4)
    elapsed_mib = fp32(B) if v3a else 0.0   # v3a (B,) int32 counter

    # -- logits + sampling -------------------------------------------------
    if v3a:
        logits_persist_mib = fp32(B * V)                 # logits_buf (fp32)
        out_buf_mib = p(B * V)                           # graph out (fp16)
        tok_hist_mib = int64(Tnew * B)                   # (max_new, B) int64
        k = min(a.top_k if 0 < a.top_k <= V else 512, V)
        sampling_mib = ((V3A_SAMPLING_BV_BYTES * B * V
                         + V3A_SAMPLING_TOPK_BK_BYTES * B * k) / MIB)
        graph_pool_mib = 0.0
        decode_transient_mib = 0.0
    else:
        logits_persist_mib = fp32(B * V)                 # lg (B,V) fp32
        out_buf_mib = 0.0
        tok_hist_mib = 0.0
        use_topp = 0 < a.top_p < 1
        sampling_mib = ((LEGACY_SAMPLING_TOPP_BV_BYTES if use_topp
                         else LEGACY_SAMPLING_TOPK_BV_BYTES) * B * V / MIB)
        graph_pool_mib = 0.0
        decode_transient_mib = (LEGACY_DECODE_BF16_UNITS * p(B * C)
                                + p(B * V))              # step fp16 logits

    # -- decode CUDA graph pool (v3a, not with deltalog) --------------------
    use_graph = v3a and not a.v3a_deltalog
    if use_graph:
        graph_pool_mib = V3A_GRAPH_BF16_UNITS * p(B * C) + p(B * V)

    # -- DeltaLog workspace (v3a only) --------------------------------------
    deltalog_mib = 0.0
    deltalog_m = 0
    if v3a and a.v3a_deltalog:
        deltalog_m = WKV_DELTALOG_TUNED_M.get((C, B), 0)
        if deltalog_m:
            # 5 x (M-1) x B x C fp16 buffers x L layers
            deltalog_mib = p(5 * (deltalog_m - 1) * B * C * L)

    # -- prefill transients (sequential B=1 per prompt) ----------------------
    prefill_state_mib = p(2 * C) + fp32(C * N)      # single-seq state (B=1)
    prefill_act_mib = PREFILL_BF16_UNITS * p(Tprompt * C)
    prefill_mib = prefill_state_mib + prefill_act_mib
    if not v3a:
        # last_logits accumulation + torch.cat double-buffer
        prefill_mib += LEGACY_PREFILL_LAST_LOGITS_BV_FP32 * fp32(B * V)

    # -- misc ---------------------------------------------------------------
    misc_mib = 64.0                                 # norm stats, RNG, emb_cache
    cublas_mib = V3A_CUBLAS_WORKSPACE_MIB if v3a else 0.0

    # torch allocator peak (== torch.cuda.max_memory_allocated per worker)
    allocated = (weights_mib + v3a_lowrank_mib + shift_mib + wkv_mib
                 + elapsed_mib + logits_persist_mib + out_buf_mib + tok_hist_mib
                 + sampling_mib + decode_transient_mib + graph_pool_mib
                 + deltalog_mib + prefill_mib + misc_mib)
    # what nvidia-smi shows: reserved pool (+ fragmentation) + CUDA context
    reserved = allocated * (1 + a.frag_pct / 100.0)
    smi = reserved + a.cuda_overhead_mib + cublas_mib
    headroom = smi * a.headroom_pct / 100.0
    total_h = smi + headroom

    # ------------------------------------------------------------------ print
    print("=" * 74)
    print(f"generate_data.py VRAM estimate (analytical, per GPU worker) "
          f"-- engine {a.engine}")
    print("=" * 74)
    print(f"config           : C={C} V={V:,} L={L} N={N} F={F} "
          f"LoRA(d/a/v/g)={Dd}/{Da}/{Dv}/{Dg}")
    print(f"run              : batch={B} max-new={Tnew} max-prompt={Tprompt} "
          f"top-p={a.top_p} top-k={a.top_k}")
    if v3a:
        print(f"v3a flags        : wkv={a.v3a_wkv} deltalog="
              f"{a.v3a_deltalog}" + (f" (M={deltalog_m})" if deltalog_m else ""))
    print(f"dtype            : fp16 ({b} B/elem) everywhere; "
          f"WKV state {'fp32' if wkv_state_bytes == 4 else 'fp16'}")
    print("-" * 74)
    hdr = f"{'component':<48}{'MiB':>13}{'GiB':>10}"
    print(hdr)
    print("-" * 74)

    def show(name, mib):
        print(f"{name:<48}{mib:>13,.1f}{mib / 1024:>10.2f}")

    show(f"model weights (fp16, {total_params:,} params)", weights_mib)
    if v3a:
        show("v3a low-rank transpose copies (LOWRANK_WEIGHT=both)",
             v3a_lowrank_mib)
    else:
        show("legacy: layer-0 v1/v2 freed (aliased to a1/a2)", v3a_lowrank_mib)
    show(f"RNN shift state L x (2, B, C) fp16", shift_mib)
    show(f"WKV state L x (B, H, N, N) "
         f"{'fp16' if wkv_state_bytes == 2 else 'fp32'}", wkv_mib)
    if v3a:
        show("logits_buf (B, V) fp32 (persistent)", logits_persist_mib)
        show("graph output (B, V) fp16 (persistent)", out_buf_mib)
        show(f"tok_hist ({Tnew}, B) int64", tok_hist_mib)
        show(f"sampler transients (fp32 copy + topk k={k})", sampling_mib)
        if use_graph:
            show("CUDA graph private pool (T=1 decode transients)",
                 graph_pool_mib)
        if deltalog_mib:
            show(f"DeltaLog workspace 5*(M-1)*B*C*fp16*L (M={deltalog_m})",
                 deltalog_mib)
    else:
        show("step logits (B, V) fp32 (persistent per step)",
             logits_persist_mib)
        show(f"sampler transients "
             f"({'top-p full-vocab sort' if use_topp else 'top-k'})",
             sampling_mib)
        show("decode step transients (~16 (B,C) fp16 + fp16 logits)",
             decode_transient_mib)
    show(f"prefill transients (B=1 state + ~{PREFILL_BF16_UNITS:.0f} "
         f"(1,T,C) fp16" + (", last_logits cat" if not v3a else ")"),
         prefill_mib)
    show("misc (norm stats, RNG, caches)", misc_mib)
    if v3a:
        show("cuBLAS/cublasLt workspaces (v3a)", cublas_mib)
    print("=" * 74)
    show("= torch allocated peak (max_memory_allocated)", allocated)
    show(f"= torch reserved (+{a.frag_pct:.1f}% allocator fragmentation)",
         reserved)
    show(f"= nvidia-smi total (+{a.cuda_overhead_mib:.0f} MiB "
         f"context/kernels)", smi)
    if a.headroom_pct > 0:
        show(f"= with {a.headroom_pct:.0f}% safety headroom", total_h)
    print("=" * 74)
    print(f"~{smi / 1024:.1f} GiB per GPU.  Multi-GPU runs (--devices auto) "
          f"spawn ONE worker\nper GPU, each holding exactly this footprint "
          f"on its own card; {B}-prompt\nblocks are processed independently "
          f"(--num-samples only sets how many\nblocks, not the peak).  "
          f"--max-new only grows tok_hist (linear, tiny).")
    print("Rule of thumb: the WKV state dominates and scales ~linearly with "
          "--batch\n(x4 for fp32io16 / legacy vs v3a fp16); weights are fixed "
          "~2.9 GiB for this\n1.5B model.  If a card OOMs, lower --batch.")

    if v3a and a.v3a_deltalog and not deltalog_m:
        print(f"WARNING: --v3a-deltalog has no tuned merge interval for "
              f"(C={C}, batch={B}); the real run would raise RuntimeError. "
              f"Use --batch one of "
              f"{sorted(bb for (c, bb) in WKV_DELTALOG_TUNED_M if c == C)} "
              f"or drop --v3a-deltalog.")

    # exact parameter count cross-check, same option as estimate_vram.py
    if a.verify_pth and a.pth and os.path.exists(a.pth):
        import torch
        sd = torch.load(a.pth, map_location="cpu", weights_only=True,
                        mmap=True)
        actual = sum(v.numel() for v in sd.values())
        ok = "OK" if actual == total_params else f"MISMATCH ({actual:,})"
        print(f"verify-pth: checkpoint numel = {actual:,} vs formula "
              f"{total_params:,} -> {ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())