# Handoff: `--mtp-arch rwkv7` support in llama.cpp — RESOLVED

**Goal of the work.** The existing RWKV-7 MTP draft support
(`docs/HANDOFF_RWKV7_MTP_LLAMACPP.md`, already implemented) only handles MTP heads
trained with `--mtp-arch mlp`. Add support for heads trained with
`--mtp-arch rwkv7` (one full RWKV-7 block as the draft head) and make the
`.pt → .gguf` conversion work for them.

**Status (updated).** Done and verified. The C++ implementation was in fact
**correct** — the divergence reported below was a bug in the *test reference*,
not in llama.cpp. `make_mtp_fixture.py::Rwkv7Head.step` (and the unrolled
reference in `scripts/verify_mtp_gguf.py`) invoked the WKV-7 recurrence with the `a`
and `b` operands **swapped** relative to the training/inference kernel. See
§4 “Resolution” for the analysis. After fixing the reference:
`test-rwkv7-mtp` is **8/8 OK** for both the rwkv7 block head and the mlp
regression head. The historical §4–§5 diagnostics are kept for context.

> **Repo reorg note.** Paths below are updated for the current layout: the
> monolithic `mtp.py` is now the `mtp/` package, every CLI tool lives in
> `scripts/`, and docs live in `docs/`. The one-off `_dbg_state.py` /
> `_dbg_variants.py` diagnostics were removed during the cleanup; their
> findings are captured in §4.2. See `MTP_README.md` for the full layout.

---

## 0. Target tree / how to build / test

```
tree    C:/Software/rwkv-related/rwkv7-autoparser-fix/llama.cpp
branch  rwkv7-autoparser-fix   (fork of ggml-org; base commit b6132bde2)
build   cmake --build build --config Release --target llama-cli
        cmake --build build --config Release --target test-rwkv7-mtp
        (MSVC 19.44, VS 17 2022 generator; CUDA present but CPU-only runs reproduce the bug)
scratch C:/Software/rwkv-related/RWKV-LM/RWKV-v7/train_temp   (reference python + test rig)
```

The working tree currently contains the full implementation (uncommitted):
`git status` → modified `src/llama-arch.{h,cpp}`, `src/llama-memory-recurrent.cpp`,
`src/llama-model.{h,cpp}`, `src/models/models.h`, `src/models/rwkv7.cpp`,
`tests/CMakeLists.txt`; untracked `tests/test-rwkv7-mtp.cpp`, `rwkv7-g1j.jinja`.

---

## 1. Reference semantics (what the head is)

`train_temp/mtp/` `RWKV7MTPModule`:

```
x      = proj( concat( LN_h(h), LN_tok(tok) ) )     # 2C -> C, hidden FIRST
x      = block.ln0(x)                               # layer-0 only pre-block norm
z      = Block(x, v_first=None)                     # layer_id == 0 semantics
logits = W_head @ z                                 # frozen backbone lm-head, no extra norm
```

`Block(args, layer_id=0).forward`:

```
x_att, _ = att(ln1(x))            # time mix, v_first = own value (no value residual)
x        = x + x_att
x        = x + ffn(ln2(x))        # channel mix
```

The head is **stateful**: the block has a WKV state `(H, N, N)` and a token
shift state `(x_prev = ln1 output, x2_prev = ln2 output)`. The training-time
parallel form and the inference reference are implemented in
`spec_decode.py::RWKV7HeadRunner._run` (pure-torch except the `wkv7sb` op).

The exact per-token WKV recurrence (also `ggml/src/ggml-cpu/ops.cpp`
`ggml_compute_forward_rwkv_wkv7_f32`, and the CUDA kernel):

```
sa[i]      = sum_j a[t,h,j] * st[h,i,j]
st'[h,i,j] = st[h,i,j]*w[t,h,j] + v[t,h,i]*k[t,h,j] + sa[i]*b[t,h,j]
y[t,h,i]   = sum_j st'[h,i,j] * r[t,h,j]
```

`llama.cpp`'s trunk graph computes `w = exp(-0.606531 * sigmoid(w0 + tanh(xw@w1)@w2))`
which is algebraically identical to the training kernel's
`exp(-exp(-softplus(-(w0+wm)) - 0.5))`. GroupNorm over heads uses `eps = 64e-5`.

The tensor mapping and the graph math were **verified numerically end-to-end**
(§3), so the per-row math and all tensor transposes/packing are correct.
Only the *state carry* is wrong.

---

## 2. What is implemented (all verified working)

### 2.1 Converter — `train_temp/scripts/convert_mtp_to_gguf.py`

* Accepts `arch ∈ {mlp, rwkv7}` (`load_mtp_payload` now returns the arch).
* rwkv7 export emits **exactly one nextn block** (`block_count = nextn_predict_layers = 1`).
  Rationale: the head is stateful, and the driver's `chain_heads` path (K>1) cannot
  roll back per-depth recurrent state → GGML_ABORT. A shared-head checkpoint's
  `num_mtp` is the *recursion depth* → maps to `--spec-draft-n-max` (default 3).
  A non-shared multi-head rwkv7 checkpoint is rejected with a clear message.
* `RWKV7_HEAD_TENSOR_MAP` maps checkpoint keys → `blk.0.nextn.*`:
  * `ln_h`/`ln_tok`/`proj` → `hnorm`/`enorm`/`eh_proj` (same as mlp)
  * `block.ln0` → `nextn.block_ln0` (new tensor name)
  * `block.ln1` → `nextn.attn_norm`, `block.ln2` → `nextn.attn_norm_2`
  * `x_r..x_g` → `nextn.time_mix_lerp_fused` packed `(6,1,1,C)`, order **r,w,k,v,a,g**
  * `w0/a0/k_k/k_a` (1,1,C) → flat `(C,)` f32; `w1/w2/a1/a2/g1/g2` **transposed** f16
    (PyTorch `(out,in)` → ggml ne `(in,out)`); `r_k` `(H,N)` → flat `(C,)`
  * `v0/v1/v2` → emitted as copies of `a0/a1/a2` (layer-0 dead slots, mirrors
    `conversion/rwkv.py` layer-0 behaviour)
  * `ln_x` → `nextn.time_mix_ln`, `ffn.x_k` → `nextn.channel_mix_lerp_k`, etc.
* Metadata for rwkv7 describes the **head block**, not the backbone: decay/iclr
  ranks from `w1/a1` shapes (128/128 for C=2560, not the backbone's 96),
  `value_residual_mix = iclr` (layer-0 alias), gate from `g1` (256),
  `wkv.head_size` from `r_k` (64), `feed_forward_length` from `ffn.key` (10240).
* Docstring documents the full mapping.

Produced (verified) artifacts:

```
train_temp/mtp_out/mtp_heads_rwkv7.gguf            821 MiB   (35 head tensors)
train_temp/mtp_out/mtp_fixture_rwkv7.bin           2.2 MiB   (8-case numeric fixture)
train_temp/mtp_out_1/mtp_heads.gguf + fixture      mlp K=1 regression artefacts
train_temp/tiny/{target,mtp,mtp_block}.gguf        synthetic end-to-end rig (C=64)
```

### 2.2 llama.cpp loader — `src/models/rwkv7.cpp` (`load_arch_tensors`)

The nextn loop probes `blk.{i}.nextn.time_mix_w0.weight`; if present it loads the
whole block tensor set into the layer's regular `attn_norm*` / `time_mix_*` /
`channel_mix_*` slots (so the graph can reuse `llm_build_rwkv7_base` verbatim) and
`block_ln0` into `llama_layer_nextn::block_ln0{,_b}` (`src/llama-model.h`).
Otherwise the old mlp layout is loaded. Mixed layouts abort.
New tensor enums/names/op-classifications added in `src/llama-arch.{h,cpp}`
(`LLM_TENSOR_NEXTN_BLOCK_LN0`, `..._ATTN_NORM`, `..._TIME_MIX_*`, `..._CHANNEL_MIX_*`).

### 2.3 llama.cpp graph — `src/models/rwkv7.cpp` (`graph_mtp`)

`graph_mtp` now derives from `llm_build_rwkv7_base` (`src/models/models.h`).
After the shared fusion (`hnorm`/`enorm`/`eh_proj`) and the optional MLP branch,
a new `head_is_block` branch runs the block:

```
cur   = ln0(fused)                      # nextn.block_ln0
rs_inp = build_rs_inp()
token_shift = build_rwkv_token_shift_load(rs_inp, ubatch, il)
att_norm = ln1(...)                     # layer.attn_norm
x_prev   = concat(att_shift, view(att_norm, skip last))
v_first  = nullptr                      # layer-0: v_first = own value
att_out  = build_rwkv7_time_mix(rs_inp, att_norm, x_prev, v_first, ubatch, il)
ffn_inp  = att_out + inpL
ffn_norm = ln2(ffn_inp)                 # layer.attn_norm_2
x_prev   = concat(ffn_shift, view(ffn_norm, skip last))
store token_shift = concat(att_norm_last, ffn_norm_last)
cur      = build_rwkv7_channel_mix(...) + ffn_inp
z        = cur                          # then masked gather -> t_h_nextn, logits = output @ z
```

This mirrors `graph::graph` (trunk) node-for-node. `il = hparams.n_layer() + nextn_layer_offset`.

### 2.4 Memory — `src/llama-model.cpp` (`create_memory`) + `src/llama-memory-recurrent.cpp`

* `create_memory`: for `ctx_type == MTP && arch == RWKV7 && n_layer() == 0`,
  `mtp_stateful_rwkv7` is true iff `layers[0].time_mix_w0 != nullptr` → the draft
  gets a **recurrent** memory. The stateless mlp draft keeps the old plain
  (empty) KV cache path (`mtp_stateless_rwkv7`).
* `llama_memory_recurrent` now iterates `hparams.n_layer_all` (instead of
  `n_layer()`) in the ctor allocation loop, `state_write_data`, `state_read_data`
  (incl. the layer-count header check) and `state_clear`, so a trunk-less draft
  (`n_layer() == 0`) gets state tensors for its NextN layers. Nullptr guards and
  the filter callbacks keep existing (mamba/RWKV/hybrid) models byte-identical —
  for every pre-existing recurrent/hybrid model `n_layer_all == n_layer()`.
* Rollback for the draft goes through the server's draft-side prompt checkpoint
  (`ctx_dft_seq_rm_type == FULL` because RWKV7 is not in `llm_arch_supports_rs_rollback`
  and `cparams.n_rs_seq` for the draft is clamped to 0). This path was observed
  working in the tiny end-to-end run (checkpoint create / restore / replay).

### 2.5 Tests

* `tests/test-rwkv7-mtp.cpp` (from the previous mlp session) is arch-agnostic: it
  decodes fixture cases into an MTP context and compares `t_h_nextn` + logits +
  argmax against a fixture.
* `train_temp/scripts/make_mtp_fixture.py` extended: auto-detects arch from the GGUF
  tensors; for rwkv7 it carries the block state **across cases in decode order**
  (`Rwkv7Head.step`).

---

## 3. Verification results (all green except §4)

```
# converter round-trip: gguf-mapped numpy vs pure-torch from the .pt
python train_temp/scripts/verify_mtp_gguf.py mtp_out/mtp_heads_rwkv7.gguf
  max |abs err| 3.444e-05   max rel 1.264e-03   argmax exact over 5 positions   -> OK
python train_temp/scripts/verify_mtp_gguf.py mtp_out_1/mtp_heads.gguf mtp_out_1/mtp_heads.pt
  max |abs err| 1.526e-05                                                      -> OK (mlp regression)

# C++ numeric parity
test-rwkv7-mtp -m mtp_out_1/mtp_heads.gguf -f mtp_out_1/mtp_fixture.bin         -> 8/8 OK (mlp)
test-rwkv7-mtp -m mtp_out/mtp_heads_rwkv7.gguf -f mtp_out/mtp_fixture_rwkv7.bin -> 1/8 OK  <-- BUG

# end-to-end draft-mtp (synthetic, random weights, so only the code paths matter)
python train_temp/scripts/make_tiny_rwkv7.py --out-dir tiny --embd 64 --layer 2
llama-cli -m tiny/target.gguf -md tiny/mtp_block.gguf --spec-type draft-mtp \
          --spec-draft-n-max 3 -p "The quick brown" -n 24 -ngl 0 -fit off -st
  -> loads, "speculative decoding will use checkpoints", creates/restores the
     draft checkpoint (draft = 0.016 MiB), drafts 3 candidates/step, exits 0
llama-cli -m tiny/target.gguf -md tiny/mtp.gguf --spec-type draft-mtp ... -st    -> still works (mlp)
```

Note: `llama-cli` stays in interactive mode unless `-st` is passed — always use `-st`
(or pipe stdin) when scripting, otherwise the process just sits there.

---

## 4. THE BUG — state does not carry across decodes

### 4.0 Resolution (2026-10-06)

**The C++ implementation is correct. The reference was wrong.**

The RWKV-7 WKV recurrence computed by the training/inference kernels
(`scripts/generate_data.py`, `spec_decode.py::RWKV7HeadRunner._run`, `src/model.py:
RWKV7_CLAMPW_CUDA`, `cuda/wkv7s_batched.cu`, and the llama.cpp graph/CUDA/CPU
kernels) is

```
sa[i]      = sum_j (-kk[j])       * st[i,j]
st'[i,j]   = st[i,j]*w[j] + v[i]*k[j] + sa[i] * (kk[j]*a[j])
```

i.e. the op is always invoked as `wkv7(r, w, k, v, a=-kk, b=kk*a)` (see
`RWKV7_CLAMPW_CUDA(r, w, k, v, -kk, kk*a)` and
`ggml_rwkv_wkv7(..., ggml_neg(kk), ggml_mul(kk, a), st)`).

`make_mtp_fixture.py::Rwkv7Head.step` instead used `sa` with `a` (the sigmoid)
and the update with `-kk*a` — the two operands are swapped. With a **zero**
start state the `sa` term vanishes, so case 0 still matches and the bug only
shows up from case 1 onwards (exactly the “state does not carry” symptom).
`scripts/verify_mtp_gguf.py` had the same swap, but it passed because it compares two
identical (both-wrong) implementations.

Diagnosis that pinned it down:

* Parsing the C++ state blob after case 0 (`llama_state_seq_get_data`) showed
the stored `(x_prev, x2_prev, kv)` matches the numpy reference to ~1e-3.
* Dumping the graph *loads* with a `cb_eval` callback showed the loaded state
at case N+1 is **bit-identical** to the stored state at case N (diff `0.0`).
  So the state carry (memory rows, `find_slot`, `rs_z`, store/load offsets) is
  not the problem.
* Feeding the dumped `(r,w,k,v,-kk,kk*a,st)` into the kernel recurrence
  reproduces the C++ `wkv_output` to f32 precision (`y` err 4.8e-7, state err
  6.0e-8). Feeding the fixture's swapped `(a,-kk*a)` does not (err ~1e-1).

Fix: `scripts/make_mtp_fixture.py` / `scripts/verify_mtp_gguf.py` now pass `a=-kk`, `b=kk*a`.
Regenerated fixture:

```
python scripts/make_mtp_fixture.py --gguf mtp_out/mtp_heads_rwkv7.gguf --out mtp_out/mtp_fixture_rwkv7.bin
python scripts/verify_mtp_gguf.py mtp_out/mtp_heads_rwkv7.gguf      # OK, max|abs err| 1.8e-05
test-rwkv7-mtp -m mtp_out/mtp_heads_rwkv7.gguf -f mtp_out/mtp_fixture_rwkv7.bin  # 8/8 OK
test-rwkv7-mtp -m mtp_out_1/mtp_heads.gguf     -f mtp_out_1/mtp_fixture.bin       # 8/8 OK (mlp)
```

No llama.cpp source needed to change for this bug. The historical sections
below are kept because they contain the (correct) state-carry analysis that
ruled out the memory path.

### 4.1 Symptom (historical)

```
test-rwkv7-mtp -m mtp_out/mtp_heads_rwkv7.gguf -f mtp_out/mtp_fixture_rwkv7.bin
fixture: C=2560 V=65536 cases=8
case 0: max|z-err|=6.145e-03  argmax 31835 vs 31835  OK        <-- fresh state: correct
case 1: max|z-err|=1.886e+00  argmax 59    vs 59     FAIL
case 2: max|z-err|=3.287e+00  argmax 26031 vs 41     FAIL
case 3: max|z-err|=5.949e+00  ...                    FAIL
case 4: 6.49e+00  case 5: 7.60e+00  case 6: 1.34e+01  case 7: 9.78e+00   FAIL
process-path decode (n_tokens=3, n_outputs=0): rc=0 OK
```

The error grows with the number of decodes → an accumulated-state mismatch, not
a per-row math error (case 0 proves the per-row math, tensor mapping, lerp
packing/order, group norm, decay transform and lm-head are all correct).

**Reproduces with `CUDA_VISIBLE_DEVICES=-1`** (identical numbers) → it is not the
CUDA wkv7 kernel; it is CPU/graph/memory state handling.

### 4.2 Diagnostic: which part of the state does the C++ carry?

`train_temp/_dbg_variants.py <draft.gguf> <outdir>` *(removed in the repo reorg — see
note at the top; kept here as the historical diagnostic record)* writes three 2-case
fixtures where both cases use the *same* `(h, tok)` and case 1's expectation is
computed with a different state variant advanced by case 0:

| fixture | expectation for case 1 | `max|z0 - z_variant|` | C++ vs fixture |
|---|---|---|
| `_dbg_both.bin`  | kv **and** token shift advanced | 18.510 | `max|z-err| = 2.929` FAIL |
| `_dbg_shift.bin` | only token shift advanced (kv zeroed) | 16.992 | `0.555` FAIL |
| `_dbg_kv.bin`    | only kv advanced (shifts zeroed) | 2.688 | `17.30` FAIL |

(identical across all three runs: the C++ case-1 output is the same, only the
expectation differs.) **The C++ output is closest to "token shift only" but does
not match any of the three** → the C++ is carrying something state-like, but not
the reference's `(x_prev, x2_prev, kv)` triple. Also
`_dbg_state.py` (two identical cases, expectation = stateless z0) fails with the
same 2.929 → the C++ is definitely not stateless.

### 4.3 What has already been checked (all consistent, so the bug is elsewhere)

* `mctx->apply()` (→ `llama_memory_recurrent::find_slot`, which sets `src0`,
  `head`, `rs_z`) runs **before** the graph is built
  (`llama_context::process_ubatch` → `mctx->apply()` then
  `graph_params(...)`/`build_model`), so the load/store indices are current.
* `llama_memory_recurrent::find_slot`: at the end, every cell in `[min,max]` gets
  `cells[i].src = i`; a self-ref makes `refcounts[i] == 1` → on the *next* decode
  `rs_z == -1` → `build_rs`'s `state_zero` view is zero-sized → **no zeroing of
  the live cell**; `s_copy(0) == src0 == 0` → the state is read from row 0.
* The graph writes the wkv state back to `get_s_l(il)` at row `kv_head == head == 0`
  and the token shift to `get_r_l(il)` at the same row (`build_rwkv_token_shift_store`).
* `build_rwkv_token_shift_load` reshapes `r_l` to `(n_embd, token_shift_count, n_seqs)`
  → offset 0 is `att_norm`, offset `n_embd` is `ffn_norm` — matches the store order
  `concat(att_norm_last, ffn_norm_last, dim=1)`.
* The per-token WKV math was re-derived from `ggml_compute_forward_rwkv_wkv7_f32`
  and matches `make_mtp_fixture.py::Rwkv7Head.step` and the torch reference
  (state layout `(H,N,N)` with `st'[h,i,j]`, `sa` uses the *previous* state,
  `y` uses the *updated* row).
* The stateful block graph is node-for-node equivalent to the trunk
  `graph::graph` block body (same builds, same order), which is known-good for
  RWKV-7 trunk models.
* The `n_layer_all` change in `llama_memory_recurrent` allocates **1 layer** of
  draft state, and `n_embd_r() = token_shift_count * n_embd`, `n_embd_s() = n_embd * wkv_head_size`.

### 4.4 Suggested next step — dump the C++ state and diff it

The state is directly observable and the layout is known. Temporarily extend
`tests/test-rwkv7-mtp.cpp` (or write a scratch program) to, **after case 0**,
call:

```cpp
size_t sz = llama_state_seq_get_size(ctx, 0);              // llama.h
std::vector<uint8_t> buf(sz);
size_t n = llama_state_seq_get_data(ctx, buf.data(), sz, 0);
fwrite(buf.data(), 1, n, fopen("cpp_state_after_case0.bin", "wb"));
```

Then parse it in Python. Layout of the blob
(`llama_context::state_seq_get_data` → `state_seq_write_data` →
`llama_memory_recurrent::state_write`):

```
u32   io_magic (0xaf143cd8)
i32   seq_id
u32   cell_count                       (== 1 for one seq)
per cell (cell_count):                 # state_write_meta
    i32 pos
    u32 n_seq_id
    i32 seq_id * n_seq_id
then state_write_data:
    u32 s_trans   (= 0)
    u32 n_layer   (= hparams.n_layer_all == 1 for the draft)
    for il in [0, n_layer):            # R tensors first
        i32 r_type ; u64 r_size_row ; <cell rows * r_size_row bytes>
    for il in [0, n_layer):            # then S tensors
        i32 s_type ; u64 s_size_row ; <cell rows * s_size_row bytes>
```

with (for C=2560, head_size=64): `n_embd_r = 5120` floats = 20480 bytes
(token shift, layout `[att_norm(2560) ; ffn_norm(2560)]`), `n_embd_s = 163840`
floats = 655360 bytes (WKV, layout `(H,N,N)` = `h*N*N + i*N + j`).

Compare against the numpy emulation after its case 0
(`make_mtp_fixture.py::Rwkv7Head.step` keeps `self.x_prev`, `self.x2_prev`,
`self.kv`; `_dbg_variants.py` *(removed)* showed how to snapshot them). That will say
immediately **which** of the three state components the C++ got wrong and by how
much, which should localise the bug to either (a) the write/store side,
(b) the read/load side, or (c) the kernel's state output view.

Prime suspects to examine once the dump points the way:

1. `llm_graph_context::build_rs` (`src/llama-graph.cpp:3496-3523`): the
   `state_zero` clearing and the `ggml_get_rows(states, s_copy_main)` read, and
   `s_copy()` (`src/llama-memory-recurrent.cpp:1346`) — especially whether the
   token-shift `r_l` and wkv `s_l` reads use the *same* row the previous decode
   wrote (check `get_head()` vs `s_copy()` vs the store offset, and that
   `llama_set_nextn_layer_offset` leaves `il == 0` for the single-head case).
2. The store views in `build_rwkv7_time_mix` / `build_rwkv_token_shift_store`:
   `ggml_cpy` into a `view_1d` of the memory tensor — the scheduler must order
   the store *after* the load in the **same** graph; the trunk has the same
   pattern, but here the graph is tiny (few nodes) and the MTP graph is a
   separate graph type — verify the node order/dependencies in the dump or with
   `GGML_SCHED_DEBUG`.
3. Whether the draft's `r_l`/`s_l` rows are the ones the graph thinks: log
   `il`, `kv_head`, `s_copy(...)`, `get_rs_z()` for both decodes (add temporary
   `fprintf`s in `build_rwkv_token_shift_load/store` and `build_rwkv7_time_mix`).

### 4.5 A/B reference (do this to avoid regressions)

Stash the implementation (`git stash`), rebuild `llama-cli`, and confirm the
mlp flows are unchanged; then restore (`git stash pop`). Both were verified:

* mlp C++ fixture test: 8/8 OK both before and after the changes.
* mlp tiny end-to-end `-md` run: works before and after.

---

## 5. Secondary (low priority) issue — standalone `-m <draft>.gguf` load

* **Before** the changes: `llama-cli -m tiny/mtp.gguf …` fails with
  `Error: the server exited before becoming ready` (the trunk graph hits the
  empty recurrent memory).
* **After**: it now reaches the scheduler and crashes with
  `ggml/src/ggml-backend.cpp:205: GGML_ASSERT(buffer) failed`, called from
  `ggml/src/ggml-backend-meta.cpp:834` (`calculate_split_state` →
  `ggml_backend_buffer_get_usage(tensor->buffer)` on a tensor whose buffer is
  still null at split-graph time). Same for the rwkv7 draft.
* This is the *degenerate* path (a trunk-less draft loaded as a target with
  `ctx_type == DEFAULT` → the empty trunk graph). The `-md` draft-mtp path is
  unaffected and works for both archs. Worth a look only if the crash is
  reachable from the supported `-md` path; otherwise document and move on.

---

## 6. Reference index

**Implementation files (this work)**

| file | what |
|---|---|
| `src/llama-arch.h` / `src/llama-arch.cpp` | +26 `LLM_TENSOR_NEXTN_*` enums, name templates, op classifications |
| `src/llama-model.h` | `llama_layer_nextn::block_ln0` / `block_ln0_b` |
| `src/models/models.h` | `llama_model_rwkv7::graph_mtp : llm_build_rwkv7_base` |
| `src/models/rwkv7.cpp` | nextn layout probe + block tensor loading; `graph_mtp` block path |
| `src/llama-model.cpp` | `create_memory`: `mtp_stateful_rwkv7` → recurrent memory |
| `src/llama-memory-recurrent.cpp` | iterate `n_layer_all` (ctor, state write/read/clear) |
| `tests/CMakeLists.txt`, `tests/test-rwkv7-mtp.cpp` | parity test (already present) |

**Scratch/tooling (`RWKV-LM/RWKV-v7/train_temp/`)**

| file | what |
|---|---|
| `scripts/convert_mtp_to_gguf.py` | converter (mlp + rwkv7), `--mtp/--pth/--vocab/--out` |
| `scripts/make_tiny_rwkv7.py` | synthetic target + mlp draft + **rwkv7 block draft** (`tiny/mtp_block.gguf`) |
| `scripts/make_mtp_fixture.py` | fixture generator, `Rwkv7Head` numpy stateful emulation (`Rwkv7Head.step`) |
| `scripts/verify_mtp_gguf.py` | gguf-vs-torch numeric round-trip, `--arch` auto (`OK` for both) |
| ~~`_dbg_variants.py`~~ | state-variant diagnostic fixtures — **removed** in the reorg; results in §4.2 |
| ~~`_dbg_state.py`~~ | stateless-expectation diagnostic — **removed** in the reorg; results in §4.2 |
| `mtp_out/mtp_heads.pt` | the real rwkv7 checkpoint (shared head, `num_mtp=3`, arch=rwkv7) |
| `mtp_out/mtp_heads_rwkv7.gguf`, `mtp_out/mtp_fixture_rwkv7.bin` | rwkv7 test artefacts |
| `mtp_out_1/mtp_heads.pt` (+ `.gguf`, fixture) | mlp K=1 regression artefacts |
| `tiny/*.gguf` | synthetic end-to-end rig |

**Reference semantics**

* `train_temp/mtp/` — `MTPModule`, `RWKV7MTPModule`, `RWKVWithMTP` (training math).
* `train_temp/src/model.py` — `Block`, `RWKV_Tmix_x070`, `RWKV_CMix_x070`.
* `train_temp/scripts/spec_decode.py` — `RWKV7HeadRunner._run` (inference reference, incl. state).
* `llama.cpp/src/models/rwkv7.cpp::graph::graph` — known-good trunk block body.
* `llama.cpp/src/models/rwkv7-base.cpp::build_rwkv7_time_mix` / `build_rwkv7_channel_mix`.
* `llama.cpp/ggml/src/ggml-cpu/ops.cpp::ggml_compute_forward_rwkv_wkv7_f32`.

**Constraints (do not violate)**

* Do not edit `common/speculative.cpp` — the driver is generic over `n_layer_nextn`.
* rwkv7 heads are single-head only (K=1); `chain_heads` cannot roll back recurrent state.
* Do not add a `GGML_ASSERT(n_layer_nextn == 1)`; keep `il = n_layer() + nextn_layer_offset`.
* The mlp path (including its f16/f32 type choices and tensor names) must keep
  passing `test-rwkv7-mtp` 8/8.
