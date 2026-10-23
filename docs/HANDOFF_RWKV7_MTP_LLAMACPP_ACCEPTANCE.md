# Handoff: RWKV-7 MTP draft-mtp acceptance-rate gap (llama.cpp vs spec_decode.py)

**Status: RESOLVED (2026-10-13, commit `6e1168d93`).** The root-cause analysis
in §3 below is *stale/wrong* — see the resolution note first. Read that before
touching `common/speculative.cpp` or the draft-mtp path.

---

## RESOLUTION (supersedes §3–§7)

The §3 "silent recurrent-rollback failure" does **not** reproduce on the
current tree. The server computes `ctx_dft_seq_rm_type == FULL` (same as the
target), so `spec_ckpt.load_dft` already rewinds the draft context every round.
I traced the draft-context state hash across `draft()`/`process()`/`accept()`
and it is clean: the rejected-draft and chain ingests are discarded before the
next persistent write. The handoff's gap was also partly an artefact of
comparing spec_decode on its own prompt against llama on a *reasoning* chat
prompt. On matched prompts the old code was already much closer than §0 claims.

Two real bugs were found and fixed (commit `6e1168d93`):

1. **Shared head used one recurrent state for all depths.** Training keeps one
   WKV state per depth (`RWKV7HeadRunner`); the K=1 shared GGUF has one block,
   and llama reused a single state recursively. Depth 0 matched the reference
   exactly, depth ≥ 1 diverged. Fix: the driver now keeps one serialized
   recurrent state per depth and uses the draft context cell as scratch
   (gated on `llama_model_is_recurrent(ctx_dft)`, single head, not shared mem).
   `process()` captures target rows, `accept()` advances every depth over the
   committed rows, `draft()` catches up lazily and runs the reference chain.
2. **Draft tokens were random, not argmax.** The draft sampler chain is
   `{TOP_K 10}`; `common_sampler_init` appends a `dist` sampler, so
   `common_sampler_sample` sampled randomly from the top-10 (and with backend
   sampling enabled the host logits buffer isn't even filled). §2's claim that
   the driver "always drafts the argmax" is wrong. Fix: take the argmax of the
   real logits directly (and detach the backend sampler for this path).

Verified: spec's `RWKV7HeadRunner` run on llama's *own* hidden states produces
`[37138, 47, 11, 46]` for `The capital of France is`, and the driver now
produces exactly that. Code-prompt acceptance went from
`(1.000, 0.879, 0.727)` to `(1.000, 0.938, 0.844)` vs the torch reference's
`(0.93, 0.857, 0.821)`. Prompt caching is supported by persisting the driver
state through `get_state`/`set_state`.

Still true from the old doc: all GGUFs on disk are `nextn_predict_layers=1`
(one shared head, `chain_heads=false`), and the `chain_heads` partial-`seq_rm`
concern only applies to a genuine multi-head GGUF (none available to test).
The §5 checkpoint design is superseded by the per-depth state buffers above.

---

## 0. TL;DR

* The user's RWKV-7 MTP heads get much better acceptance in `scripts/spec_decode.py`
  than in llama.cpp's `--spec-type draft-mtp` (greedy: ~0.88 / 0.81 / 0.85
  per depth vs llama's `0.738 / 0.500 / 0.378` per slot; tokens/round 2.21 vs
  1.62 of MTP-accepted drafts).
* Part of the gap was **metrics + setup** (conditional vs unconditional
  tables, sampling params, and spec_decode's extra free target pick — see §2).
  But a large part is a **real state-corruption bug** in the driver:
  RWKV-7's recurrent memory **cannot roll back tokens** (`n_rs_seq = 0`,
  partial `seq_rm` fails *silently*), so the driver's catch-up/chain re-ingests
  stale and duplicate rows into the draft head's state every round. The head
  states are systematically polluted → acceptance is depressed at every depth.
* A full fix was attempted (checkpoint-based advance/chain mirroring
  `scripts/spec_decode.py`) and got ~90% of the way (a K=1 recursive flow that ran and
  stopped failing) but was **reverted at the user's request** after repeated
  crashes (one unresolved segfault). See §5 for the design and §6 for every
  pitfall hit, so the next attempt doesn't rediscover them.

## 1. Environment

* Fork: `C:/Software/rwkv-related/rwkv7-autoparser-fix/llama.cpp`
  (branch `rwkv7-autoparser-fix`, the user's rwkv7-MTP work on top of upstream
  master; working tree has the user's own uncommitted MTP implementation in
  `src/llama-arch.*`, `src/llama-memory-recurrent.cpp`, `src/llama-model.*`,
  `src/models/rwkv7.cpp`, `src/models/models.h`, `tests/`).
* Target GGUF: `RWKV7-G1k-1.5B-GGUF .../rwkv7-g1k-1.5b-20260930-ctx25600-FP16.gguf`
  (n_embd 2048, 1.5B).
* Draft GGUF: `train_temp/mtp_out_ob_1.5b_lk/mtp_heads.gguf`
  **IMPORTANT: this file is K=1** (`rwkv7.nextn_predict_layers = 1`,
  `block_count = 1`, 37 tensors — one shared head, written once). The user
  trains with `--num-mtp 3` in spec_decode.py, but the GGUF contains a single
  head; llama.cpp reaches depth 3 via **recursive reuse** (`chain_heads = false`
  because `n_mtp_layers == 1`), not via the multi-head `chain_heads` path.
  Do not assume `chain_heads` is active for this file.
* Reference implementation: `train_temp/scripts/spec_decode.py`
  (`RWKV7HeadRunner`: per-depth WKV states, `snapshot()/restore()`,
  `advance()`, `draft_chain()`). It is the training-exact scheme.
* Server test command (greedy):
  ```
  llama-server -m <target>.gguf -md train_temp/mtp_out_ob_1.5b_lk/mtp_heads.gguf \
      --spec-type draft-mtp -ngl 99 -c 4096 --temp 0 --top-k 0 -np 1 \
      --spec-draft-n-max 3 -sm none --main-gpu 0 -lv 4
  ```
  Then a `/v1/chat/completions` request; the per-slot acceptance is in
  `acc per pos = (...)` (trace level) and `draft acceptance = ...` (INF).
* Per-depth Python acceptance: `python scripts/spec_decode.py --pth
  rwkv7-g1k-1.5b-20260930-ctx25600.pth --config config-1.5b.json --mtp
  mtp_out_ob_1.5b_lk/mtp_heads.pt --num-mtp 3 --mtp-arch rwkv7 --max-new 2048
  --num-prompts 1 --device cuda:0 --greedy 1`

## 2. What is NOT a bug (metrics/setup differences)

* **Metric mismatch.** spec_decode's per-depth table is *conditional* for
  depths ≥ 2 (`att[j]` only increments when the chain reached that depth);
  llama's `acc per pos` is unconditional over all rounds. Compare:
  llama `(0.738, 0.500, 0.378)` vs spec unconditioned `(0.880, 0.716, 0.612)`
  (spec cond: `0.814`, `0.854`). The MTP-accepted drafts per round:
  llama 1.62 (265/164) vs spec 2.21 (148/67).
* **Sampling params.** `--greedy 0` in spec_decode is temp 0.8 / top-p 0.9 /
  no top-k; the user's sampled llama run was temp 1.0 / top-k 20 / **top-p 0.95
  (default, not set)** → 0.465 overall. Lower temp on the llama side closes
  most of the sampled-mode gap on its own. Greedy (`--temp 0` vs `--greedy 1`)
  is the clean comparison.
* **Draft-picking policy.** The draft-mtp driver always drafts the *argmax* of
  the draft logits (hardcoded `{TOP_K 10}` sampler, takes `cur_p->data[0].id`
  in `draft()`), regardless of `--temp`; spec_decode samples its drafts.
  Greedy drafting is if anything favorable, but the two setups are not
  identical unless you edit the driver's draft sampler.
* **Free target picks.** Both schemes commit the target's own pick at the
  round boundary free of charge; per-round structure is otherwise the same
  (2 free + m gated; llama's `mean len = 1 + accepted/round`).
* **Different text.** Greedy streams diverge between ggml and torch
  (fp16 backbone numerics), so per-text acceptance comparisons are loose.
  For a hard number, teacher-force the *same* token stream (extend
  `scripts/eval_mtp.py`).

## 3. The real bug: silent recurrent-rollback failure

* RWKV-7 is **not** in `llm_arch_supports_rs_rollback` (`src/llama-arch.cpp`
  ~1175), so `cparams.n_rs_seq` is clamped to 0 (`src/llama-context.cpp`
  ~106) and `llama_memory_recurrent::seq_rm` partial rollback
  (`src/llama-memory-recurrent.cpp:165`) always returns **false** for
  `0 < p0 <= cell.pos`. The `wkv7` kernel also never writes per-token state
  snapshots (only final state), which is *why* RWKV7 is excluded there.
* The driver's `common_speculative_impl_draft_mtp` calls partial `seq_rm`
  in `process()` (per-head catch-up) and `draft()` (per-chain-step) and
  **ignores the return value**. Every round, therefore:
  * the catch-up re-ingests the whole verify batch (4 rows) on top of a state
    that still contains the previous round's chain rows and rejected drafts;
  * the chain re-ingests rows 0..n on top of that;
  * nothing is ever rewound → per round the head ingests ~7 rows for
    ≤ 5 committed tokens, with duplicated and stale-rejected pairs.
* Result: the head's WKV/token-shift state is a smeared accumulator →
  acceptance is depressed at **every** depth (not only deep ones). The
  server falls back to whole-state checkpoints for the *target*
  (`ctx_tgt_seq_rm_type == FULL` → `spec_ckpt`), but the driver's internal
  draft-memory bookkeeping is what corrupts the draft head states.
* Verified empirically: making only the *capture/advance* path clean (and
  leaving heads ≥ 1 partially wrong) lifted slot 1 from 0.738 → 0.782 while
  slots 2/3 stayed at 0.500/0.35 — consistent with per-head pollution.

## 4. The correct semantics (from spec_decode.py)

Per round, per depth k (stream end = last committed position `p`; the fresh
target pick `c_{p+1}` is treated as committed):

* **advance** — ingest the newly committed pairs under every head:
  row at stream position `P` pairs `(z_{k-1}[P-1-k], c_P)` (depth 0 pairs
  `(h_{P-1}, c_P)`). spec does this in `advance()` with `zbufs[k-1]` rows;
  its per-depth watermark afterwards is `j <= p' - k - 1`.
* **chain** — ingest `k` tentative rows per depth (positions `pos0+1..pos0+k`,
  tokens `d_1..d_k`), hiddens chained from the previous head/step; depth k's
  last row produces draft `d_{k+1}`. spec does this in `draft_chain()`
  (`j = p-k .. p`, `k+1` rows because its `advance` watermark stops one earlier —
  the llama mapping is equivalent; see the verified identity
  `j_k(t) = pos0 + t - k - 1`).
* **snapshot/restore** — the chain's ingests are tentative; the next round
  must roll back to the post-advance watermark. spec uses per-depth state
  clones; llama must use explicit state checkpoints
  (`llama_state_seq_get_data/set_data` — the same API the server's
  `spec_ckpt` already uses, proven to work on this memory).

Because `llama_state_seq_*` restores the whole sequence state, the driver
never needs partial `seq_rm` at all.

## 5. The attempted fix (reverted — re-implement along these lines)

All changes were in `common/speculative.cpp` (+ 2 small edits in
`src/models/rwkv7.cpp` graph_mtp). Gate everything on a new member:

```cpp
ckpt_mode = llama_model_is_recurrent(llama_get_model(ctx_dft));
```

(recurrent draft memory ⟹ rollback impossible; keep the legacy path for
non-recurrent KV drafts untouched — their `seq_rm` works).

New per-seq state: `cap_tok/cap_pos/cap_h` (captured target rows),
`ingested_upto`, `last_pos0`, `dirty`, `has_save`, `state_save`,
`bonus_z[head]` (head's z at its bonus row), `ing_last_z[head]` (head's z at
its last *ingested committed* row), `d1_tok/d1_p`.

Flow (per seq):

* `process()` → **capture only**: append `(token, pos, h)` rows from the
  target's dense `t_h_nextn` to the cap; also populate `verify_h`/`verify_h_rows`
  (accept() needs them) and set `pending_h` to the last row's h (bootstrap;
  accept() overwrites it with `h_{pos0-1}`).
* `accept(m)` → `chain_restore()` (discard chain ingests) →
  `chain_ingest(last_pos0 + m)` (ingest the accepted rows; their captured
  tokens are committed by definition) → prune cap to `[ingested_upto, acc_end]`.
* `draft()` → `chain_restore()` → lazy `chain_ingest(pos0 - 1)` (plain
  non-drafted rounds leave rows in the cap) → **bonus ingest**: the pair
  `(pending_h, id_last) @ pos0` under every head, **sample d1 from head 0's
  logits at this row**, then `chain_save()` (clean watermark) → chain:
  * `chain_heads` (K>1): head k decodes the k rows drafted so far, row r ≥ 1
    embd = `z_prev[r-1]` (prev head's z at table row r−1), row 0 embd =
    `bonus_z[k-1]`; produces `d_{k+1}` at its last row.
  * K=1 recursive: each step decodes ONE row `(head's own previous z,
    d_step) @ pos0+step` at layer offset 0 (the DeepSeek-style recursive
    reuse; exact for depth 0/1, approximate beyond — the K=1 GGUF cannot do
    better without per-depth states).
  * set `dirty = true` after any chain decode.
* `advance_decode(seq, head, toks, poss, rows, h_rows, z_prev, z_first,
  logits_row, z_cur)` helper:
  * head 0: `embd(r) = h_rows[r]` = target h at `pos_r - 1`
  * head k, r=0: `z_first` — must be **`ing_last_z[k-1]`** (the previous
    ingest boundary; NOT `bonus_z[k-1]`, which is one round stale whenever
    the last round accepted ≥ 1 draft — this was bug #6 below)
  * head k, r≥1: `z_prev[r-1]` = head k−1's z at table row r−1
  * row pos 0 is never ingestable (no `h_{-1}`): filter `pos >= head+1` per head.
* **Draft ctx must use unmasked nextn** (`llama_set_embeddings_nextn(ctx_dft,
  true, /*masked*/ false)`): the advance needs *every* row's `t_h_nextn`, and
  the draft context's **backend sampling rejects >1 output row per sequence**
  (`src/llama-context.cpp:1752`), so decodes must carry 0-or-1 `logits=true`
  rows (`logits_row` param: −1 for advance decodes, the sampled row otherwise).
  This requires a `src/models/rwkv7.cpp` `graph_mtp` change: publish
  `res->t_h_nextn` **densely** (before the out_ids gather, like
  `deepseek2.cpp:699`), gather only when `cparams.embeddings_nextn_masked`,
  and **only build `inp_out_ids` when masked** — an unused registered input
  tensor has no buffer and crashes the scheduler
  (`GGML_ASSERT(buffer) failed`, `ggml-backend.cpp:205`).

## 6. Pitfalls hit during the attempt (all were real crashes/failures)

1. **Zero-arg `SPC_WRN("...")`** fails to compile under MSVC (the macro passes
   `__VA_ARGS__` into a printf with a trailing comma). Use `SPC_WRN("%s", "...")`.
2. **Member init gating**: the per-seq state vectors must be initialized under
   `ckpt_mode`, not `chain_heads` — for the K=1 GGUF `chain_heads` is false and
   the first capture segfaulted on empty vectors.
3. **`verify_h` population**: the capture early-return must still fill
   `verify_h`/`verify_h_rows`, or accept()'s `pending_h` bookkeeping breaks.
4. **pos-0 row**: skip it in the ingest table (`continue`), never `break` the
   whole table because of it (it permanently has no `h_{-1}`).
5. **`chain_bonus_and_save` must set `ingested_upto = pos0`** — otherwise the
   next advance's table re-starts at the already-ingested bonus position and
   llama-batch's consecutive-position check fails
   (`X = 22, Y = 22 ... Y = X + 1` → `failed to initialize batch` → ret −1).
6. **Bonus-row `z_first`** must be `ing_last_z[k-1]` (see §5); `bonus_z[k-1]`
   is one round stale once any draft is accepted. With the bug, heads ≥ 1 get
   a wrong hidden on one row per round and stay degraded.
7. **UNRESOLVED — segfault in a ~47-row advance `llama_decode`** on the draft
   context (long 91-token prompt; the short 22-token prompt worked end-to-end).
   Facts: ctor-time `llama_n_batch(ctx_dft)` printed 2048 and
   `llama_n_ubatch` 512; smaller tables (1/13/21/22/23 rows) decoded fine;
   crash consistently around row 46–48 of the batch build/decode. No debugger
   is installed on the machine (no cdb/windbg/gdb). Suspects, in order:
   (a) the impl's `batch` is sized from **ctor-time** `llama_n_batch` which may
   differ from the final fitted value — re-derive or re-init the batch lazily;
   (b) chunk the advance (≤16 rows per decode) — the design is already
   chunk-safe if `z_first = ing_last_z[k-1]` per chunk; (c) something in the
   recurrent memory/graph at larger row counts. Debug with unbuffered
   `fprintf(stderr, ...); fflush(stderr);` markers — llama's own logger is
   buffered and loses everything at a hard crash.
8. **K>1 (`chain_heads`) path was never tested** — the tested GGUF is K=1.
   The same scheme applies; the z-chaining multi-head loop was written but
   not exercised. Also note: with `ckpt_mode`, the graph_mtp unmasked change
   and ≤1-outputs rule apply to it as well.
9. **Do not break the legacy (non-recurrent draft) path** — the classic
   draft-model impls and the non-recurrent draft_mtp case rely on
   `seq_rm`-based KV rollback, which works for them. Gate on `ckpt_mode`.
10. **llama's logger buffers** — use raw `fprintf(stderr)+fflush` for crash
    bisection; taskkill /F loses all buffered log lines.
11. If you kill/rebuild while a server is running, the link fails with
    `LNK1104: cannot open llama-common.dll` — kill `llama-server.exe` first
    (`taskkill //IM llama-server.exe //F` in git-bash; single `/` is eaten).

## 7. Expected outcome once correct

* Slot acceptance should approach spec_decode's greedy numbers
  (≈0.88 / 0.81 / 0.85 conditional; 0.88/0.72/0.61 unconditional) modulo
  text/numerics (fp16 ggml vs torch) — head 0 was *already* exact before the
  fix, so ~0.74→0.88 was the pre-fix slot-1 gap to recover.
* Tokens/round should go from 2.62 toward ~3.6+ (m ≈ 2.2 + 1 bonus + 1 pick).
* Throughput note: the attempted version was *slower* per token (79 vs
  111 t/s) because of many tiny decodes + 2 state copies/round. The state
  copy is ~2–3 MB (3 layers: 2×2048 f32 token-shift + 2048×64 f32 WKV per
  head) — consider batching the advance across heads, or larger advance
  chunks, before judging quality-by-speed.

## 8. Current repository state (after revert)

* `common/speculative.cpp` = HEAD (upstream driver, unmodified).
* `src/models/rwkv7.cpp` = the user's own version (my two graph_mtp edits
  reverted; `git diff` shows only the user's MTP work).
* All other modified files (`llama-arch.*`, `llama-memory-recurrent.cpp`,
  `llama-model.*`, `models.h`, `tests/CMakeLists.txt`, `tests/test-rwkv7-mtp.cpp`)
  are the user's own pre-existing work — do not revert those.
* `llama-server.exe` rebuilt from the reverted sources; no stray processes.
* The two small `rwkv7.cpp` edits you will need again are described in §5
  (dense `t_h_nextn` + masked-gated gather + conditional `build_inp_out_ids`).
