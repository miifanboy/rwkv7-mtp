# Variable-length MTP head for a frozen RWKV-7 — status & usage

Retrofit implementation of the handoff. **The full pipeline now runs on this
machine**: CUDA kernels compile, the G1 checkpoint loads, the backbone is frozen,
and a one-batch overfit passes. The only thing missing to train for real is a
training corpus (`data/train.bin`).

## Repository layout

The MTP retrofit is a subproject of the upstream RWKV-7 `train_temp` tree.  The
upstream training demo (`train.py`, `src/`, `demo-*.sh`) is untouched;
everything MTP-specific is grouped as follows:

```
train_temp/
├── mtp/                  # MTP library package (was the monolithic mtp.py)
│   ├── env.py            # RWKV_* env, CUDA/MSVC build fixes, PL shim
│   ├── tensors.py        # _shift_left and pure tensor helpers
│   ├── heads/            # MTP head architectures
│   │   ├── mlp.py        #   MTPModule        (--mtp-arch mlp)
│   │   └── rwkv7.py      #   RWKV7MTPModule   (--mtp-arch rwkv7)
│   ├── losses/           # training objectives (chunked, memory-bounded)
│   │   ├── common.py     #   CE_CHUNK, _ce
│   │   ├── ce.py         #   fused head+CE (_head_ce)
│   │   ├── distill.py    #   top-k self-distillation (_distill_ce)
│   │   ├── lk.py         #   LK / acceptance losses (_lk_loss)
│   │   └── mtp.py        #   generic logits-in mtp_loss
│   ├── backbone.py       # frozen RWKV-7 + chained heads (RWKVWithMTP)
│   ├── checkpoint.py     # .pth loading, mtp_heads.pt save/load/inspect
│   └── cli.py            # `python -m mtp shift|inspect`
├── scripts/              # CLI entry points (run as `python scripts/<name>.py`)
│   ├── train_mtp.py      # train the heads on a frozen backbone
│   ├── eval_mtp.py       # per-depth acceptance table
│   ├── spec_decode.py    # speculative draft/verify loop
│   ├── generate_data.py  # self-distilled corpus (--dataset tulu|ultradata)
│   ├── extract_val.py    # materialise the --val-fraction hold-out
│   ├── convert_mtp_to_gguf.py / verify_mtp_gguf.py
│   ├── make_tiny_rwkv7.py / make_mtp_fixture.py
│   └── parse_train_log.py / plot_metrics.py / compare_accept.py
├── tests/                # CPU + CUDA-gated suites
│   ├── test_mtp_logic.py
│   ├── test_mtp_train.py
│   ├── test_rwkv7_head.py
│   └── test_lk_loss.py
├── docs/                 # design notes / handoffs
│   ├── HANDOFF_RWKV7_MTP_LLAMACPP.md
│   ├── HANDOFF_RWKV7_MTP_BLOCK_HEAD.md
│   ├── HANDOFF_RWKV7_MTP_LLAMACPP_ACCEPTANCE.md
│   ├── fastmtp.md
│   └── lk-loss.md
├── cuda/                 # upstream kernels + MTP additions (wkv7s_batched, lk_loss)
├── src/                  # upstream RWKV-7 model
└── data*/ mtp_out*/ ...  # datasets and training outputs (unchanged)
```

### Key modules

| Module | Purpose |
|--------|---------|
| `mtp/heads/mlp.py` | `MTPModule` -- position-local MLP head (`--mtp-arch mlp`, the original). |
| `mtp/heads/rwkv7.py` | `RWKV7MTPModule` -- one full RWKV-7 block (`--mtp-arch rwkv7`). |
| `mtp/losses/ce.py` | Fused head-projection + CE, the training path (`_head_ce`). |
| `mtp/losses/distill.py` | Top-k self-distillation KL (`_distill_ce`). |
| `mtp/losses/lk.py` | LK acceptance objectives, hybrid / `-log alpha` (`docs/lk-loss.md`). |
| `mtp/checkpoint.py` | `build_frozen_model`, save/load MTP heads (arch-tagged + verified). |
| `mtp/backbone.py` | `RWKVWithMTP` -- frozen backbone + chained heads. |
| `scripts/train_mtp.py` | Plain PyTorch training loop (AdamW + cosine/warmup). Saves `mtp_out/mtp_heads.pt`. |
| `scripts/generate_data.py` | Self-distilled generator; `--dataset tulu` (default) or `ultradata` (the openbmb no_think mix). |
| `scripts/convert_mtp_to_gguf.py` | Packs `mtp_heads.pt` (+ frozen `token_embd`/`output`) into a llama.cpp draft GGUF. |
| `scripts/verify_mtp_gguf.py` | GGUF metadata + tensor map + numeric round-trip. |
| `tests/test_mtp_logic.py` | CPU-only tests (shift, shapes, loss, masking, freeze, grad flow, chunked/fused CE, checkpoint arch round-trip). |
| `tests/test_mtp_train.py` | One-batch overfit on the real frozen backbone (`--mtp-arch mlp|rwkv7`). |
| `tests/test_rwkv7_head.py` | CUDA-gated tests for the rwkv7 head (grads, recursion, recurrent runner parity/rollback). |
| `tests/test_lk_loss.py` | CPU + CUDA-kernel checks for the LK losses. |
| `cuda/lk_loss.{cpp,cu}` | Fused full-vocab LK kernels. |
| `cuda_compat.h`, `cuda/wkv7s_batched.*` | CUDA-13 compat shim and a batch-aware copy of the single-step kernel. |

> The legacy CLI forms are gone: `python mtp.py shift` is now
> `python -m mtp shift`, and every tool moved under `scripts/`.

## Verified on this machine (RTX 4070 SUPER, 12 GB)

```
$ python tests/test_mtp_logic.py    # CPU, no backbone
shift ok
PASS  test_shift
PASS  test_mtp_module_shapes  (24,960 params/head)
PASS  test_mtp_loss_perfect  (loss=0.0000)
PASS  test_ignore_index_masking  (2.9853)
PASS  test_memory_efficient_loss_matches_reference  (3.617084)
PASS  test_grad_flow_and_freeze  (loss 4.0620 -> 1.4494)
PASS  test_ce_chunk_invariance  (default chunk=256)
PASS  test_fused_head_ce_matches_reference  (value + dz)
PASS  test_fused_head_ce_weight_grad_and_freeze
PASS  test_fused_head_ce_end_to_end_grad_flow
All MTP logic tests passed.

$ python scripts/train_mtp.py --check-only --device cuda:0 --ctx-len 4096
[mtp] resized 256 low-rank params to checkpoint shapes
[train_mtp] MTP params: 118,010,880 (3.849% of 3,066,076,160)
[train_mtp] trainable tensors: 27
[train_mtp] --check-only: OK

$ python tests/test_mtp_train.py --ctx-len 512 --steps 60 --lr 2e-3
step   0  total=11.610  main=12.051 mtp_1=11.398 mtp_2=11.405 mtp_3=11.334
step  59  total=4.082   main=12.051 mtp_1=0.000  mtp_2=0.000  mtp_3=0.000
combined loss 11.610 -> 4.082  (64.8% drop)
ONE-BATCH OVERFIT OK
```

Backbone = 2.948 B frozen params, MTP = 118 M trainable (K=3), only
`mtp_heads.*` receives gradients.

## Environment (already applied)

`torch` was replaced with a CUDA build in the global pyenv:

```bash
pip install --index-url https://download.pytorch.org/whl/cu130 \
            --extra-index-url https://pypi.org/simple \
            "torch==2.10.0+cu130" "torchvision==0.25.0+cu130"
```

cu130 (not cu128) because the only installed toolkit is CUDA 13.2; `exllamav3`
still requires only `torch==2.10.0`, and torchvision 0.25.0 replaces the
mismatched 0.28.0.

Three CUDA-13 / MSVC build fixes are applied automatically by `mtp/env.py` before
importing `src.model` (`_prepare_build_env` / `_patch_cpp_extension_flags`), so
plain `python scripts/train_mtp.py ...` works without manual env exports:

1. `CL=/Zc:preprocessor` — CUDA 13's CCCL headers reject MSVC's traditional
   preprocessor.
2. `-std=c++20` on CUDA sources — avoids a torch 2.10 / MSVC
   `C2872: 'std': ambiguous symbol` failure in nvcc's host pass.
3. `-include cuda_compat.h` — restores `atomicAdd(float2*, float2)`, which CUDA
   13 only defines for `__CUDA_ARCH__ >= 900` (the 4070 is sm_89).

`CUDA_HOME` and `TORCH_CUDA_ARCH_LIST` are auto-detected if unset.
`RWKV_JIT_ON` is forced to `1` (see quirks below).

## Quirks in the provided `src/model.py` (worked around, not edited)

* **`RWKV_JIT_ON` must be 1.** The file defines a module-level `_forward_op`
  five times; with JIT off the last definition wins and every kernel wrapper
  calls the wrong function. `train.py` uses JIT on, which is why this is latent.
* **Low-rank dims.** `config.json` uses explicit dims
  (`a/decay_low_rank_dim=96`, `v_low_rank_dim=64`, `gate_low_rank_dim=320`), but
  `RWKV_Tmix_x070.__init__` derives them from formulas (128/128/96/256 at
  C=2560). `build_frozen_model` resizes the 256 mismatched parameters to the
  checkpoint shapes before loading (the tmix matmuls are shape-agnostic, so this
  is exactly equivalent to constructing with the config dims).
* `load_state_dict` rejects shape mismatches even with `assign=True`, hence the
  resize step.

## Usage

```bash
# 1. quick checks
python -m mtp shift
python -m mtp inspect ./rwkv7-g1j-2.9b-20260831-ctx16384.pth
python scripts/train_mtp.py --check-only --device cuda:0

# 2. train the MTP heads (K is just --num-mtp; backbone never changes).
#    --mtp-arch selects the head: 'mlp' (position-local, the original) or
#    'rwkv7' (one full RWKV-7 block; recurrent at inference, needs extra VRAM
#    for the WKV backward state and prefers a lower lr).  Auto-detected from
#    the checkpoint when resuming with --load-mtp.
python scripts/train_mtp.py \
  --data data/train.bin --data-dtype uint16 \
  --pth rwkv7-g1k-2.9b-20260930-ctx25600.pth --config config.json \
  --num-mtp 3 --ctx-len 4096 --batch-size 1 --grad-acc 4 \
  --mtp-arch rwkv7 --shared-head \
  --lr 1e-4 --steps 2500 --decay 0.8 --device cuda:0 --out mtp_out \
  --val-fraction 0.02 --val-every 100

# 3. acceptance
python scripts/eval_mtp.py \
  --data data/val.bin --data-dtype uint16 \
  --pth rwkv7-g1k-2.9b-20260930-ctx25600.pth --config config.json \
  --mtp mtp_out/mtp_heads.pt --num-mtp 3 --ctx-len 2048 --device cuda:0
```

## Validation (`--val-fraction`)

`--val-fraction F` holds out the **tail** of `--data` as validation, so one file
serves as both train and val with no overlap and nothing duplicated on disk.

* `train_mtp.py` supports a half-open token range (`BinDataset(..., start, stop)`)
  and the two ranges are disjoint by construction.
* The cut is snapped forward to the first token-0 (EOS). `generate_data.py`
  appends exactly one EOS per conversation, so train and val never share a
  half-document.
* `--val-data <file>` still works and takes precedence; `--val-fraction 0`
  (the default) disables validation entirely.

The val line reports every component, so it lines up with the per-depth train
numbers:

```
step     99 VAL loss 6.1234 main 0.5012 vd1=2.4120 vd2=2.8760 vd3=3.1904
```

Validation always includes the main head regardless of `--skip-main-ce`:
`forward_loss` normalises by a weight that counts the main slot either way, so
reporting a total that omits it from the numerator would just be scaled by
`K/(K+1)`. It also costs only ~3.5% of a forward pass, and it is the readout
that tells you whether the held-out text is in distribution.

### Recovering the hold-out as a file

The split is in-file, so no `val.bin` is written. `eval_mtp.py` and
`spec_decode.py` both want a path, so materialise it:

```bash
python scripts/extract_val.py --data data/train.bin --val-fraction 0.05 --out data/val.bin
# [extract_val] data/train.bin[14,912,897:15,696,698] -> data/val.bin
# [extract_val] 783,801 tokens (4.99% of 15,696,698), cut_at_eos=True

python scripts/eval_mtp.py --data data/val.bin --data-dtype uint16 \
  --mtp mtp_out/mtp_heads.pt --num-mtp 1 --ctx-len 2048 --device cuda:0
```

The cut is **deterministic** — `val_cut()` uses no RNG, seed, time, or
filesystem ordering, only `filesize // itemsize`, `--val-fraction`, and the
first token-0 (EOS) at/after the cut point. Re-running with the same
`--val-fraction` reproduces the same range byte-for-byte. Both the resolved
range and `val_fraction` are recorded in `train_log.txt` and in the
`mtp_heads.pt` meta (`val_range`, `val_fraction`), so a checkpoint always pins
down its own validation set. `--start/--stop` override the computation.

## Graphable metrics (`metrics.jsonl`)

Alongside the human-readable `train_log.txt`, `train_mtp.py` appends one JSON
object per logged event to `<out>/metrics.jsonl`:

```json
{"t": "2026-10-04T13:04:20", "type": "train", "src": "live", "run": "mtp_out_cont",
 "run_idx": 1, "step": 10, "steps": 32000, "lr": 1.09e-05, "total": 11.0128,
 "main": 0.4665, "mtp1": 16.5538, "mtp2": 16.3557, "mtp3": 16.2743,
 "kl1": 10.6683, "kl2": 10.5362, "kl3": 10.4835, "gnorm": 2.547,
 "ktok_s": 4.3, "tokens_seen": 123520}
```

* `type` is `train`, `val`, or `run_start` (run_start carries the full args
  dict); `run_idx` is the 1-based index of the `NEW RUN` header in
  `train_log.txt`, so records from different eras of the same out-dir never
  collide.
* `kl1..N` only appear when `--distill-topk > 0`; `main` is omitted under
  `--skip-main-ce` (train records only — val always includes it).
* Records written before this existed can be backfilled from the text log with
  `parse_train_log.py` (idempotent — it rebuilds the whole file and merges any
  live records back in, so it can be re-run at any time):

  ```bash
  python scripts/parse_train_log.py --log mtp_out_cont/train_log.txt
  ```

* Plot it with `plot_metrics.py` (train curves = faint raw + bold EMA, val
  points overlaid, restart boundaries dashed):

  ```bash
  python scripts/plot_metrics.py --metrics mtp_out_cont/metrics.jsonl
  python scripts/plot_metrics.py --metrics mtp_out_cont/metrics.jsonl --x tokens --smooth 0.95
  python scripts/plot_metrics.py --metrics mtp_out_cont/metrics.jsonl --split-runs --show
  ```

  By default the crash-resume sections inside one `train_log.txt` are merged
  into a single continuous curve, deduplicating re-executed steps (a resume
  re-runs steps after the last checkpoint; the later record wins).
  `--split-runs` draws each `NEW RUN` section separately instead.

## `--skip-main-ce`

The main-head CE is the *frozen* backbone head applied to the *frozen* backbone
hidden state, so `ce_main.grad_fn is None`: it has no gradient at all and is
pure cost. Measured 3.5% of a step. `forward_loss(include_main=False)` skips it;
because `total_w` still counts its weight, the MTP gradient is **bit-identical**
(verified: `torch.equal` on every head parameter, max delta 0.000e+00). Only the
reported `loss` changes (it drops the constant `ce_main / total_w` offset), so
the field is omitted from the log line rather than printed as 0.0000.

It is worth keeping **on** for a first run: `main` is how you notice that the
self-distilled corpus has drifted out of the backbone's distribution.

## LK losses (`--mtp-loss lk-hybrid` / `--mtp-loss lk-alpha`)

See `docs/lk-loss.md` for the mathematics. The depth loss is replaced by an objective
that directly optimizes the speculative-decoding acceptance rate

```
alpha = sum_x min(p(x), q(x)) = 1 - TV(p, q)
```

where `q` is the depth's draft distribution and `p` the frozen backbone teacher
at the aligned position. The teacher probabilities are recomputed over the
FULL vocabulary (the top-k distillation table is not usable: TV/alpha need the
draft's full support), with the same chunked recompute-in-backward memory
discipline as the fused CE — nothing `(n, V)`-sized is retained.

- `lk-hybrid`: `lam*KL(p||q) + (1-lam)*TV(p,q)` with the adaptive schedule
  `lam = exp(-eta * mean(alpha))` (`--lk-eta`, paper default 3.0). Early on
  (low alpha) KL dominates and stabilizes training; as alpha rises TV takes
  over and directly optimizes acceptance. Trust-region interpretation in
  docs/lk-loss.md §3.1.
- `lk-alpha`: `-log(alpha)` — TV optimization with adaptive `1/alpha` gradient
  scaling; degenerates to plain NLL when `p` is a point mass.

Requires `--distill-topk 0` and no `--kl-only` (they are replaced, not
combined). Log/val lines gain `a1..aK` / `va1..vaK` fields: the mean alpha per
depth, i.e. the direct deployment metric the loss optimizes (also emitted to
`metrics.jsonl` as `alpha1..alphaK`). Everything else — checkpoint format,
eval/spec-decode, the CE and distill paths — is unchanged.

Speed: with the fused CUDA kernels (`cuda/lk_loss.cu`, built automatically on
the first LK CUDA call and cached; `RWKV_LK_CUDA=0` forces the eager fallback)
LK is now ~7% FASTER than the top-k `--kl-only` path (7.2K vs 6.8K tok/s on
the 12 GB card at ctx 4096) — the fused block-per-row sweep over the bf16 GEMM
outputs is cheaper than the distill path's eager full-vocab softmax + scatter.
Without the kernels the eager LK path runs at ~4.5K tok/s (~30% slower than
`kl-only`). The teacher logits table is built once per micro-step and shared
across depths (`--lk-teacher-cache auto`, ~0.5 GiB bf16 at ctx 4096).
`lk-alpha` and `lk-hybrid` cost the same. Exactness is unchanged: the kernels
are verified bit-identical (hybrid) / fp32-noise-identical (lk-alpha) against
the eager path by `tests/test_lk_loss.py`.

## Shift convention

`targets[i] = x[i+1]`. Main head predicts `x[i+1]`. MTP depth `k` (0-indexed):
input token `x[i+k+1]`, target `x[i+k+2]` (depth 0 predicts 2 ahead, depth `k`
predicts `k+2` ahead). Heads are chained (depth `k` consumes depth `k-1`'s
hidden). `tests/test_mtp_logic.py::test_shift` derives and asserts this from the
handoff's `targets[i+k+1]` rule.

> The handoff's passing mention of `_shift_left(targets, k-1)` for the *input*
> conflicts with its own target rule and would skip the immediately-preceding
> token; this implementation uses the self-consistent `k+1` input.

## K at inference vs training

There are **two different ways** to get a deep draft, and they are easy to
conflate. `K` (the number of trained heads) only controls one of them.

**Recursive reuse — one head, any depth.** Apply the same head repeatedly, each
step feeding it its own output hidden state plus the token it just sampled:

```
h_0 = backbone hidden        -> z_0 = MTP_0(h_0, emb(x))        -> draft 1
h_1 = z_0                    -> z_1 = MTP_0(h_1, emb(draft 1))  -> draft 2
...
```

This needs **no extra training** and works with `--num-mtp 1`. `K=1` therefore
already supports `--spec-draft-n-max N` for **any** `N` — that is exactly how
DeepSeek-V3-style single-layer MTP and Qwen3.5 are used, and why the llama.cpp
default for `--spec-draft-n-max` is `3` rather than `1`. Only the *first* step is
in distribution; later steps are extrapolation, so acceptance decays with depth.

**Chained heads — `K` specialised heads, capped at `K`.** Train `K` heads so each
depth gets its own head specialised for it (Step-3.5 style; the llama.cpp driver
calls this `chain_heads`). Head `k` is trained on the true depth-`k` input, so
acceptance at depth `k` is better than recursion would give. The trade is that
the driver then caps the draft at `min(n_max, K)` — there is no head `K` to walk
past, so `K` becomes a hard ceiling. `K=3` means you can never draft 4.

So pick `K` by what you are optimising:

| | draft depth | cost | quality at depth |
|---|---|---|---|
| `K=1` + recursion | unbounded | 1 head (39M) | decays with depth |
| `K=n` + chain | at most `n` | `n` heads | better, and bounded |

For this project (one trained head, `--num-mtp 1`) you get unbounded depth for
free, and `--spec-draft-n-max` is the knob that trades draft cost against
acceptance. Training more heads is a **quality** decision at a fixed depth, not a
prerequisite for reaching that depth.

> Consequence worth stating plainly: the earlier framing "`K` = the largest
> lookahead you will ever want" is true only in the chained regime. With `K=1`
> you can always draft deeper than you trained for — it just accepts less often.

## Memory (12 GB, 4070 SUPER)

Backbone is 5.5 GiB bf16 and runs under `torch.no_grad()`.

`forward_loss` goes through `_head_ce` (`_FusedHeadCE` in `mtp/losses/ce.py`), which
fuses the shared head projection with the cross-entropy: **no `(B,T,V)` tensor
is ever materialised**, and the projection is recomputed chunk-wise in the
backward pass instead of saving a `(B,T,V)` fp32 log-softmax. That matters
because the obvious implementation retains ~1 GiB *per MTP depth*: at T=4096,
V=65536 the fp32 logits are 1 GiB, `F.cross_entropy` saves another 1 GiB, and
all of it stays alive until `backward()`. `del logits_k` does not reclaim it --
the autograd graph, not the Python name, is what pins the tensor.

Measured (`torch.cuda.max_memory_allocated`), `--num-mtp 3`, forward + backward:

| ctx-len | peak allocated | peak reserved |
|---------|----------------|---------------|
| 2048    | 6.3 GiB        | 6.6 GiB       |
| 4096    | 6.6 GiB        | 6.8 GiB       |

So `--ctx-len 4096 --num-mtp 3` fits with ~5 GiB to spare (the old logits-based
path OOM'd at exactly these settings). What still grows with T is the backbone
forward and the `(B,T,C)` hidden states, not the loss.

Chunking is controlled by `RWKV_MTP_CE_CHUNK` (read at import, default 256 rows).
Transient memory is `chunk * vocab_size * 4` bytes (~64 MB at V=65536) and is
independent of context length; larger chunks trade memory for fewer launches.

Loss values are bit-identical to the old path. Gradients differ by ~1.3% in
norm (cosine 0.99992), purely because both paths run the projection in bf16 and
round in a different order.

## Self-distilled data (`generate_data.py`)

Streams `allenai/tulu-3-sft-mixture` (filtering the broken
`ai2-adapt-dev/flan_v2_converted` source), renders each conversation as a G1
prompt (`System:/User:/Assistant:`), prefills through the frozen backbone in RNN
mode, then samples the model's own assistant replies in batched lockstep.
Generation is fp16 (the single-step kernel is `at::Half`).  Output is a flat
`uint16` `.bin` of `prompt + response + [EOS]` plus a `.json` sidecar.

By default (`--devices auto`) it runs one worker process per CUDA GPU.  Work is
handed out as blocks (`--batch` prompts) through a shared queue: whichever GPU
finishes first pulls the next block, so a fast GPU never idles behind a slow
one (e.g. 4070 SUPER + 3060 12GB).  Blocks are written to the .bin in block
order and each block is seeded by its index, so scheduling does not change the
dataset (the legacy backend is bit-reproducible across identical GPUs; the
faster3a_2607 backend uses fp16 atomics in some FFN kernels, so its reruns may
differ in occasional low-bit token flips while totals stay stable; mixed GPU
architectures always differ in fp16 low bits).

Two backends (`--engine`):

* `v3a` (default): the `faster3a_2607/` engine — fused CUDA kernels, CUDA-graph
  decode, GPU-side token feedback, top-k/top-p sampler.  Measured on the
  4070 SUPER + 3060 rig with this 2.9B model at `--batch 256 --max-new 256`:
  ~16.5 s per 256-seq block on the 4070 SUPER and ~33 s on the 3060, i.e.
  ~7.4k tok/s aggregate (~3.4x the legacy backend).  `--v3a-deltalog` adds the
  engine's DeltaLog WKV path (+3-7% decode, needs `--batch` in its tuned
  table: 8/16/32/64/256/512 for this model).
* `legacy`: the original in-process `BatchedRWKV7` (wkv7sb kernel).

```bash
# quick smoke (one prompt, printed)
python scripts/generate_data.py --smoke --max-new 128

# the 50k prototype dataset: both GPUs, v3a engine (~2 h -> ~35 min)
python scripts/generate_data.py --num-samples 50000 --max-new 128 --batch 256 \
    --out data/tulu_selfgen.bin

# with DeltaLog (needs --batch 256)
python scripts/generate_data.py --num-samples 50000 --max-new 128 --batch 256 \
    --v3a-deltalog --out data/tulu_selfgen.bin

# single GPU only / original backend
python scripts/generate_data.py --devices single --device cuda:1 --num-samples 50000 ...
python scripts/generate_data.py --engine legacy --devices single --device cuda:1 ...

# then train the heads on it (adjust steps as desired)
python scripts/train_mtp.py --data data/tulu_selfgen.bin --data-dtype uint16 \
    --num-mtp 3 --ctx-len 4096 --batch-size 1 --grad-acc 8 \
    --lr 2e-4 --steps 5000 --decay 0.8 --device cuda:0 --out mtp_out
```

Measured throughput: ~7.4k tok/s aggregate with the default v3a backend at
batch 256 on the 4070 SUPER + 3060 rig (~3.4x the legacy backend's ~2.2k tok/s;
~4.9k tok/s on the 4070 SUPER alone, ~2.4k on the 3060 alone).  Legacy
backend: ~1.5k tok/s at batch 64 (50k samples ≈ 20M tokens ≈ 1.9 h).
Batched generation is numerically verified against solo generation
(`wkv7s_batched.cu` fixes the upstream B=1-only state indexing).

## Speculative decoding (`spec_decode.py`)

Greedy draft→verify→commit over the RNN state.  The verification forward over
the `K+1` drafts is non-mutating; when every draft is accepted the bonus token
is fed from the verification state, otherwise the accepted prefix is re-run.

```bash
python scripts/spec_decode.py --mtp mtp_out/mtp_heads.pt --num-prompts 4 \
    --max-new 128 --device cuda:0
```

It prints baseline vs speculative tokens/s, mean tokens/step, and per-depth
conditional acceptance (`mtp_1` = first drafted token, i.e. 2 ahead).

### Prototype result (heads trained only 300 steps on 102k self-distilled tokens)

```
baseline 43 tok/s | speculative 44 tok/s (1.02x) | mean 2.35 tokens/step
mtp_1 0.247   mtp_2 0.259   mtp_3 0.571   (conditional on reaching the depth)
```

That is deliberately an undertrained check: the main head scores CE 0.59 on the
self-generated text (the model is confident in its own outputs), and the MTP
heads dropped 11.35 → 2.31.  Expect the depth-1 band (0.55–0.75) after the full
50k dataset and a real training run.

## Remaining

1. `python scripts/generate_data.py --num-samples 50000 ...` (~35 min with the default
   v3a backend on both GPUs; ~1.9 h on one 4070 SUPER with `--engine legacy`).
2. `python scripts/train_mtp.py --data data/tulu_selfgen.bin ... --steps 5000 ...`.
3. `python scripts/eval_mtp.py` / `python scripts/spec_decode.py` to report acceptance and
   speed.

## llama.cpp / GGUF

`--spec-type draft-mtp` does not work out of the box, but the gap is **much
smaller than it looks**: llama.cpp already ships the entire `draft-mtp` driver
(`common/speculative.cpp`, generic over `nextn_predict_layers`) and the `nextn`
staging API (`src/llama-ext.h`).  What is missing is only an arch that can load a
trunk-less MTP file and a graph that evaluates the head.

### The draft GGUF (done)

```bash
python scripts/convert_mtp_to_gguf.py \
    --mtp mtp_out/mtp_heads.pt \
    --pth rwkv7-g1k-2.9b-20260930-ctx25600.pth \
    --out mtp_out/mtp_heads.gguf
python scripts/verify_mtp_gguf.py          # metadata + tensor map + numeric round-trip
```

751 MB for K=1: 79 MB of head weights plus the two frozen backbone tensors the
head contracts to (`token_embd`, `output` — 320 MB each, bf16→f16 is lossless
here).  Converts any K.

The key trick is

```
rwkv7.block_count          = K
rwkv7.nextn_predict_layers = K     ->  n_layer() = 0
```

so every block in the file is a NextN block.  Two existing mechanisms then fire
with no code change: `common_speculative_types_from_gguf()` probes for
`blk.{block_count-1}.nextn.eh_proj.weight` and infers `draft-mtp` (so
`--spec-type` is optional), and the driver reads `n_mtp_layers = K` from
`nextn_predict_layers`.

Verified numerically against the real checkpoint: head forward matches
`MTPModule` to **1.5e-05**, the duplicated `output`/`token_embd` match the
backbone to **3e-08**, and draft logits reproduce the frozen head's argmax
exactly.

### What still has to be written in C++

A small, well-precedented change — see `docs/HANDOFF_RWKV7_MTP_LLAMACPP.md` for the
full spec (file-by-file diffs, the exact graph, risks, test plan).  Summary:

| file | change |
|---|---|
| `src/llama-arch.h` / `.cpp` | 3 new tensor enums + name templates (`mlp_norm`, `mlp_up`, `mlp_down`) |
| `src/llama-model.h` | extend `llama_layer_nextn` (MLP + the two LayerNorm biases) |
| `src/models/rwkv7.cpp` | trunk-less (`mtp_only`) loader mode; `graph_mtp`; dispatch on `LLM_GRAPH_TYPE_DECODER_MTP`; **~3 lines on the target side** (`res->t_h_nextn = cur`) |

`common/speculative.cpp` needs **no changes at all**.

Two traps, both silent (they degrade acceptance instead of crashing): the head's
norms are full `nn.LayerNorm` with bias (`LLM_NORM`, **not** `LLM_NORM_RMS`,
unlike every other MTP in the tree), and the concat order is
`[hidden, token]`, the opposite of DeepSeek/GLM4.

Acceptance target once built: **≈2.6 tokens/step** (draft acceptance ≈0.63),
the number `eval_mtp.py` measures offline on the held-out split.

### `--spec-draft-n-max`

Works out of the box and **any `n` is legitimate** — that is the whole point of an
MTP head. Modern MTP modules are expected to serve any draft depth even though
the drafter was only trained for one step, and llama.cpp is built that way:

* The driver names the single-head path as a first-class mode, not a degraded
  one: *"neither (qwen35 / qwen35moe): a single trained MTP head"*
  (`common/speculative.cpp:1343`). Qwen3.5 ships **one** MTP layer and drafts
  with `--spec-draft-n-max 3` **by default**.
* `chain_heads` (one *trained* head per step, Step-3.5 style) is an alternative
  specialisation, **not** a prerequisite for deeper drafting. Our K=1 file needs
  no K>1 support to draft `n=8`.
* Only *block* drafters clamp `n` — the llama.cpp docs say `--spec-draft-n-max`
  "is clamped to the draft model's trained block size" for `draft-dflash` and
  `draft-dspark`, and say nothing of the kind for `draft-mtp`. That clamp exists
  because a block draft emits a fixed-size block per graph; MTP has no such
  structure.

So with our K=1 head the flag maps onto recursion: each step feeds the head its
own output hidden state plus the token it just sampled. It needs no KV, no state
snapshot and no re-init, because the head is stateless — it is the same functional
form the chained training recurrence uses (`h_prev = z`), applied repeatedly. And
it is **exact**: speculative decoding accepts against the *target's* distribution,
so an out-of-distribution draft can only lower acceptance, never corrupt output.

The one real trap is on the implementation side: llama.cpp's other MTP graphs
(DeepSeek/GLM4, BailingMoE3) all `GGML_ASSERT(n_layer_nextn == 1)`. Copying that
assert would `abort()` on a K>1 draft. The handoff (§3.3, §7-T4b) has the details
and a sweep to characterise the acceptance/speed curve.

The implementation target is `C:/Software/rwkv-related/rwkv7-autoparser-fix/llama.cpp`
(branch `rwkv7-autoparser-fix`, commit `b6132bde2`, a jinja/autoparser fork on a
recent master base). It already has the `draft-mtp` driver, the `nextn` staging
API, and the GLM4 / BailingMoE3 MTP graphs to copy — nothing needs back-porting.
Every `file:line` anchor in the handoff was re-verified against that exact commit.
