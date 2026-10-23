"""
estimate_vram.py -- analytical VRAM estimate for scripts/train_mtp.py.

Takes the SAME command line as train_mtp.py, never builds a model, never
touches CUDA, and prints a per-component VRAM breakdown derived from closed
formulas over (config dims, --ctx-len, --batch-size, --num-mtp, ...).

The formulas mirror the actual code paths:

  * weights     : frozen backbone (bf16) + trainable MTP heads (bf16).
                  Backbone dims come from --config (the G1 checkpoints resize
                  the low-rank params to the config's *_low_rank_dim /
                  intermediate_size, see mtp/checkpoint.py
                  _resize_params_to_checkpoint).  MTP-head low-rank dims come
                  from src/model.py's heuristic formulas (the head block is NOT
                  resized -- its keys are absent from the backbone .pth).
  * optimizer   : AdamW keeps 2 moments (exp_avg, exp_avg_sq) per trainable
                  tensor, in the param dtype (bf16 here).
  * activations : only the MTP-head graphs are retained for backward (the
                  backbone runs under no_grad; only its EAGLE-3 taps survive).
                  All K=num-mtp depth graphs live simultaneously because the
                  per-depth losses are summed before the single
                  backward().  Per depth the RWKV-7 head block retains ~36
                  (B,T,C)-shaped bf16 tensors (tmix mix6 / kk-pre / clampw /
                  lnx saved tensors, ffn mixed+act, chain norms) plus the WKV
                  backward state the clampw kernel saves for its backward:
                  s (B,H,T/16,N,N) fp32 = 4*(B,T,C) fp32 and sa (B,T,H,N)
                  fp32 = 1*(B,T,C) fp32.  grad-acc does NOT change VRAM (one
                  micro-step graph at a time); batch-size and ctx-len do.
  * loss        : the fused CE / LK losses never materialise (B,T,V); only a
                  CE_CHUNK-row transient remains (chunk * V fp32 buffers).
                  LK mode may add the full-vocab teacher logits table
                  (B*T,V) bf16 -- one-shot per micro-step, kept only when the
                  auto rule (<= 1 GiB) allows it, exactly like
                  mtp/losses/lk.py _teacher_logits.

Peak = train-step peak.  Validation steps run under no_grad and are strictly
smaller.  Add ~5-10% allocator fragmentation in practice.

Example
-------
    python scripts/estimate_vram.py \
        --data data-ob-1.5b/train_75.bin --data-dtype uint16 \
        --pth rwkv7-g1k-1.5b-20260930-ctx25600.pth --config config-1.5b.json \
        --num-mtp 3 --shared-head --mtp-arch rwkv7 --warmup 200 \
        --val-chunks 64 --skip-main-ce --mtp-loss lk-hybrid --lk-eta 3.0 \
        --ctx-len 3088 --batch-size 4 --grad-acc 2 --lr 2e-4 \
        --lr-final 5e-6 --steps 10000 --decay 0.8 \
        --out mtp_out_ob_1.5b_eagle2 --device cuda:0 \
        --val-fraction 0.1 --val-every 250
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

MIB = 2**20

# ---------------------------------------------------------------------------
# Tunable constants of the model (documented, keep in sync with src/model.py)
# ---------------------------------------------------------------------------

HEAD_SIZE_DEFAULT = 64          # src/model.py: HEAD_SIZE (config head_dim)
CE_CHUNK_DEFAULT = 256          # mtp/losses/common.py CE_CHUNK

# Per-MTP-depth retained-activation coefficients, in units of a (B, T, C)
# bf16 tensor (1 unit = elemsize*B*T*C bytes).  Raw count of the tensors each
# custom op / autograd Function saves for backward in one rwkv7-head block is
# ~36 (tok emb | ln cat | proj out | ln0/ln2 | mix6 7 | r,k,v,w,g | kk-pre 3
# + inv_d | wkv y | lnx out | z | head ln_out | ln2 out | ffn mixed+act | ffn
# out); the calibrated value below is what matches the measured graph
# (bit-exact weight count + measured forward-graph peak on a 1.5B / ctx 3088
# / batch 4 / K=3 run: 6719 MiB).  The clampw WKV backward state (fp32 units
# below) is counted separately and exactly.
RWKV7_HEAD_BF16_UNITS = 31.0
# WKV backward state saved by the clampw kernel, in (B,T,C)-sized fp32 units:
#   s  = (B, H, T/16, N, N) fp32 = 4*(B,T,C) fp32 elements
#   sa = (B, T, H, N)       fp32 = 1*(B,T,C) fp32 element
RWKV7_HEAD_WKV_FP32_UNITS = 5.0

# Backbone-side tensors that survive into the trainable graph (EAGLE-3 taps
# h_early / h_mid / post-ln_out hidden + the fc_fuse input cat + fused h0):
FUSION_BF16_UNITS = 7.0

# Transient loss-chunk buffers (fused CE / LK recompute): only CE_CHUNK rows
# of (n, V) are alive inside one chunk iteration (fp32 logits + softmax).
LOSS_CHUNK_FP32_COPIES = 2.0

# Transient gradient buffers during backward (clampw allocates dr..db, plus
# lnx/kk-pre/cmix chain grads); these partially overlap saved tensors that
# are freed as they are consumed, so the calibrated value is well below the
# raw ~20 units.
BACKWARD_BF16_UNITS = 8.0

# nvidia-smi overhead beyond torch's caching-allocator reserved pool
# (CUDA context, kernel module images, cuBLAS/cuDNN workspaces).
CUDA_CONTEXT_MIB_DEFAULT = 150.0
# torch reserved > allocated (caching-allocator block rounding/fragmentation).
FRAGMENTATION_PCT_DEFAULT = 4.3


# ---------------------------------------------------------------------------
# args
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Estimate train_mtp.py VRAM usage (analytical, no GPU)")
    # training-relevant (affect VRAM or the printed report)
    ap.add_argument("--config", required=True)
    ap.add_argument("--num-mtp", type=int, default=3)
    ap.add_argument("--shared-head", action="store_true")
    ap.add_argument("--mtp-arch", default="rwkv7", choices=["rwkv7", "mlp"])
    ap.add_argument("--eagle3-fusion", dest="eagle3_fusion",
                    action="store_true", default=None)
    ap.add_argument("--no-eagle3-fusion", dest="eagle3_fusion",
                    action="store_false")
    ap.add_argument("--ctx-len", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-acc", type=int, default=8)
    ap.add_argument("--mtp-loss", default="ce",
                    choices=["ce", "lk-hybrid", "lk-alpha"])
    ap.add_argument("--lk-teacher-cache", default="auto",
                    choices=["auto", "on", "off"])
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float32"])
    ap.add_argument("--distill-topk", type=int, default=0)
    ap.add_argument("--skip-main-ce", action="store_true")
    # accepted for CLI compatibility, do not influence VRAM
    ap.add_argument("--data", default="")
    ap.add_argument("--data-dtype", default="uint16")
    ap.add_argument("--pth", default="")
    ap.add_argument("--load-mtp", default="")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr-final", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--decay", type=float, default=0.8)
    ap.add_argument("--val-fraction", type=float, default=0.0)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--val-chunks", type=int, default=16)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lk-eta", type=float, default=3.0)
    ap.add_argument("--distill-lambda", type=float, default=0.5)
    ap.add_argument("--kl-only", action="store_true")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--ignore-index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    # estimator knobs
    ap.add_argument("--verify-pth", action="store_true",
                    help="load the .pth on CPU and cross-check the backbone "
                         "parameter count (slow-ish, exact)")
    ap.add_argument("--cuda-overhead-mib", type=float,
                    default=CUDA_CONTEXT_MIB_DEFAULT,
                    help="nvidia-smi overhead beyond the torch allocator pool")
    ap.add_argument("--frag-pct", type=float,
                    default=FRAGMENTATION_PCT_DEFAULT,
                    help="torch reserved-over-allocated fragmentation, %%")
    ap.add_argument("--headroom-pct", type=float, default=0.0,
                    help="extra safety headroom added to the final total, %%")
    ap.add_argument("--ce-chunk", type=int, default=CE_CHUNK_DEFAULT,
                    help="RWKV_MTP_CE_CHUNK value used at train time")
    args, _unknown = ap.parse_known_args(argv)
    # eagle3 default: on for fresh rwkv7 runs (train_mtp.py logic)
    if args.eagle3_fusion is None:
        args.eagle3_fusion = args.mtp_arch == "rwkv7"
    return args


# ---------------------------------------------------------------------------
# dims
# ---------------------------------------------------------------------------

def load_dims(config_path: str):
    """(C, V, L, F, Dd, Da, Dv, Dg, head_size) from a config.json.

    Supports the HF-style G1 configs (hidden_size/num_hidden_layers/
    intermediate_size/*_low_rank_dim) and raw RWKV configs (n_embd/n_layer/...).
    The BACKBONE low-rank dims come from the config (the checkpoint resizes the
    constructed params to these shapes).
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    C = int(cfg.get("hidden_size", cfg.get("n_embd")))
    V = int(cfg["vocab_size"])
    L = int(cfg.get("num_hidden_layers", cfg.get("n_layer")))
    F = int(cfg.get("intermediate_size",
                    int(C * 3.5) // 32 * 32))
    Dd = int(cfg.get("decay_low_rank_dim", _lora(2.5, C)))
    Da = int(cfg.get("a_low_rank_dim", _lora(2.5, C)))
    Dv = int(cfg.get("v_low_rank_dim", _lora(1.7, C)))
    Dg = int(cfg.get("gate_low_rank_dim", _lora(5.0, C)))
    H = int(cfg.get("head_dim", cfg.get("head_size", HEAD_SIZE_DEFAULT)))
    return C, V, L, F, Dd, Da, Dv, Dg, H


def _lora(mult: float, C: int) -> int:
    """src/model.py heuristic: D = max(32, round(mult*sqrt(C)/32)*32)."""
    return max(32, int(round(mult * math.sqrt(C) / 32)) * 32)


def head_lora_dims(C: int):
    """Low-rank dims of the MTP head's block (NOT resized from the checkpoint,
    so src/model.py's heuristic formulas apply regardless of config)."""
    return _lora(2.5, C), _lora(2.5, C), _lora(1.7, C), _lora(5.0, C)


# ---------------------------------------------------------------------------
# parameter formulas (exact, verified against the G1 checkpoints)
# ---------------------------------------------------------------------------

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


def mtp_head_params(C, F, arch, eagle3, num_depths):
    """Params of ONE MTP head module.  num_depths = K for --shared-head, else 1.

    The head block's low-rank dims follow src/model.py's formulas (its keys are
    not in the backbone checkpoint, so no resize happens).  The head block is
    a layer-0 block, so att.v0/v1/v2 are ALIASED onto a0/a1/a2
    (mtp/heads/rwkv7.py) and must be counted once -- their numel is
    subtracted."""
    Hd, Ha, Hv, Hg = head_lora_dims(C)
    if arch == "mlp":
        hidden = max(32, (2 * C) // 32 * 32)
        n = 2 * C + 2 * C + 2 * C * C          # ln_h, ln_tok, proj
        n += 2 * C + C * hidden + hidden * C   # mlp: ln, up, down
        return n
    n = 2 * C + 2 * C + 2 * C * C              # ln_h, ln_tok, proj
    if eagle3:
        n += 2 * C + 2 * C + 3 * C * C         # ln_early, ln_mid, fc_fuse
    n += block_params(C, F, Hd, Ha, Hv, Hg, ln0=True)
    n -= C + 2 * C * Hv                        # aliased att.v0/v1/v2
    n += 2 * C * max(0, num_depths - 1)        # ln_h_extra[k]
    n += 2 * C                                 # ln_out
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
    C, V, L, F, Dd, Da, Dv, Dg, H = load_dims(a.config)
    bytes_per_param = 2 if a.dtype == "bfloat16" else 4
    B = max(1, a.batch_size)
    T = a.ctx_len
    K = max(1, a.num_mtp)
    n_modules = 1 if a.shared_head else K
    num_depths = K if a.shared_head else 1
    lse = lambda n: n * bytes_per_param / MIB
    # one (B,T,C) activation "unit", in bytes (param dtype per --dtype)
    row = lambda n: B * T * C * n * bytes_per_param
    row_fp32 = lambda n: B * T * C * n * 4   # fp32 WKV backward state units

    use_lk = a.mtp_loss in ("lk-hybrid", "lk-alpha")

    # -- weights ----------------------------------------------------------
    bb = backbone_params(C, V, L, F, Dd, Da, Dv, Dg)
    head_one = mtp_head_params(C, F, a.mtp_arch, a.eagle3_fusion, num_depths)
    heads = head_one * n_modules
    backbone_mib = lse(bb)
    head_mib = lse(heads)

    # -- optimizer / grads (trainable = MTP heads only) ---------------------
    grads_mib = head_mib
    # AdamW: exp_avg + exp_avg_sq, same dtype as the params (bf16 here)
    optim_mib = 2 * head_mib

    # -- persistent activations (per micro-step, all alive until backward) --
    if a.mtp_arch == "rwkv7":
        per_depth_units = RWKV7_HEAD_BF16_UNITS
        per_depth_wkv = RWKV7_HEAD_WKV_FP32_UNITS
        bw_units = BACKWARD_BF16_UNITS
    else:
        per_depth_units = 8.0      # mlp head: tok, cat, proj, ln, mlp hidden/silu, z
        per_depth_wkv = 0.0
        bw_units = 2.0
    act_fusion_mib = FUSION_BF16_UNITS * row(1) / MIB
    act_heads_mib = K * (per_depth_units * row(1)
                         + per_depth_wkv * row_fp32(1)) / MIB
    # transient grad buffers during the (single) backward pass
    act_backward_mib = bw_units * row(1) / MIB

    # -- LK teacher logits cache -------------------------------------------
    cache_bytes = B * T * V * bytes_per_param
    use_cache = use_lk and (
        a.lk_teacher_cache == "on"
        or (a.lk_teacher_cache == "auto" and cache_bytes <= (1 << 30)))
    teacher_cache_mib = (cache_bytes / MIB) if use_cache else 0.0

    # -- transient loss-chunk buffers ---------------------------------------
    loss_chunk_mib = (LOSS_CHUNK_FP32_COPIES * a.ce_chunk * V * 4) / MIB

    # -- batch + misc -------------------------------------------------------
    batch_mib = (2 * B * T * 8) / MIB                     # x, y int64
    misc_mib = 64.0                                       # norms stats, clip, rng...

    # torch allocator peak (== torch.cuda.max_memory_allocated at train time)
    allocated = (backbone_mib + head_mib + grads_mib + optim_mib
                 + act_fusion_mib + act_heads_mib + act_backward_mib
                 + teacher_cache_mib + loss_chunk_mib + batch_mib + misc_mib)
    # what nvidia-smi shows: reserved pool (+ fragmentation) + CUDA context
    reserved = allocated * (1 + a.frag_pct / 100.0)
    smi = reserved + a.cuda_overhead_mib
    headroom = smi * a.headroom_pct / 100.0
    total_h = smi + headroom

    print("=" * 74)
    print("train_mtp.py VRAM estimate (analytical)")
    print("=" * 74)
    print(f"config           : C={C} V={V:,} L={L} F={F} "
          f"LoRA(d/a/v/g)={Dd}/{Da}/{Dv}/{Dg} head={H}")
    print(f"dtype            : {a.dtype} ({bytes_per_param} B/elem)")
    print(f"batch            : {B} x {T} ctx "
          f"({B * T:,} tokens/micro-step; grad-acc {a.grad_acc} "
          f"does not change VRAM)")
    print(f"MTP              : arch={a.mtp_arch} K={K} "
          f"modules={'1 (shared)' if a.shared_head else str(K)} "
          f"eagle3={a.eagle3_fusion}")
    print(f"loss             : {a.mtp_loss}"
          + (f"  [teacher cache {'ON' if use_cache else 'off'}]"
             if use_lk else ""))
    print("-" * 74)
    hdr = f"{'component':<44}{'MiB':>13}{'GiB':>10}"
    print(hdr)
    print("-" * 74)

    def show(name, mib):
        print(f"{name:<44}{mib:>13,.1f}{mib / 1024:>10.2f}")

    show(f"backbone weights (frozen, {L} layers)", backbone_mib)
    show(f"MTP head weights (trainable, {n_modules} mod)",
         head_mib)
    show("MTP gradients (bf16)", grads_mib)
    show("AdamW moments 2x (exp_avg + exp_avg_sq)", optim_mib)
    show(f"activations: EAGLE-3 taps + fusion ({FUSION_BF16_UNITS:.0f} units)",
         act_fusion_mib)
    show(f"activations: {K} MTP depth graphs "
         f"(~{per_depth_units:.0f} bf16 units"
         + (f" + {per_depth_wkv:.0f} fp32 WKV-state units)"
            if per_depth_wkv else ")"),
         act_heads_mib)
    if use_cache:
        show("LK teacher logits table (B*T, V) bf16", teacher_cache_mib)
    show(f"loss chunk transients (~{a.ce_chunk} x V fp32 x"
         f"{LOSS_CHUNK_FP32_COPIES:.0f})", loss_chunk_mib)
    show("batch x/y (int64) + misc", batch_mib + misc_mib)
    print("=" * 74)
    show("= torch allocated peak (max_memory_allocated)", allocated)
    show(f"= torch reserved (+{a.frag_pct:.1f}% allocator fragmentation)",
         reserved)
    show(f"= nvidia-smi total (+{a.cuda_overhead_mib:.0f} MiB context/kernels)",
         smi)
    if a.headroom_pct > 0:
        show(f"= with {a.headroom_pct:.0f}% safety headroom", total_h)
    print("=" * 74)
    print(f"~{smi / 1024:.1f} GiB as seen by nvidia-smi.  Validation steps "
          f"(no_grad) are\nstrictly smaller; save steps stream to CPU and add "
          f"no GPU peak.")
    print("Rule of thumb: peak scales ~linearly with --batch-size and "
          "--ctx-len;\n--grad-acc trades throughput, not VRAM.")

    # exact parameter counts for reference
    print("-" * 74)
    print(f"backbone params: {bb:,}  |  MTP head params: {heads:,} "
          f"({heads / (bb + heads) * 100:.3f}% of model)")
    print(f"per-depth activation: "
          f"{(per_depth_units * row(1) + per_depth_wkv * row_fp32(1)) / MIB:.0f} MiB"
          f"  |  per-micro-step activations: "
          f"{(act_fusion_mib + act_heads_mib + act_backward_mib):,.0f} MiB")

    if a.verify_pth and a.pth and os.path.exists(a.pth):
        import torch
        sd = torch.load(a.pth, map_location="cpu", weights_only=True,
                        mmap=True)
        actual = sum(v.numel() for v in sd.values())
        ok = "OK" if actual == bb else f"MISMATCH ({actual:,})"
        print(f"verify-pth: checkpoint numel = {actual:,} vs formula "
              f"{bb:,} -> {ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())