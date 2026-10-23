# Handoff: RWKV-7 MTP draft head for llama.cpp (`--spec-type draft-mtp`)

**Goal.** Make this work:

```bash
llama-cli -m rwkv7-2.9b.gguf -md mtp_heads.gguf --spec-type draft-mtp \
          -ngl 99 -c 4096 -n 256
```

where `mtp_heads.gguf` is produced by `scripts/convert_mtp_to_gguf.py` (done — see §2).

**Scope.** llama.cpp already ships the whole `draft-mtp` *driver*
(`common/speculative.cpp`, ~400 lines) and a `nextn` staging API
(`src/llama-ext.h`). What is missing is (a) an **arch that can load a
trunk-less MTP draft file**, and (b) a **graph that evaluates the head**. Both
are small, well-precedented changes. Do **not** touch `common/speculative.cpp`.

**Target tree.**

```
C:/Software/rwkv-related/rwkv7-autoparser-fix/llama.cpp
  branch   rwkv7-autoparser-fix
  commit   b6132bde2  "common : always define add_generation_prompt in jinja context"
  remote   https://github.com/miifanboy/llama.cpp.git
```

This is the user's fork (a jinja / chat-autoparser workstream, recently rebased
onto ggml-org master tip — `#29448`, `#29468`, `#29443`). **Every file:line anchor
below was re-verified against a checkout of this exact commit**; do not trust the
numbers against a different tree (the same anchors in `C:/Software/llama.cpp`
@ `7798007a2` and `rwkv-related/adaptive-mtp/llama.cpp` are shifted by tens of
lines).

It already contains everything the plan depends on: the `draft-mtp` driver, the
`nextn` staging API (`src/llama-ext.h`), `llm_graph_input_embd_h`,
`LLM_TENSOR_NEXTN_*`, `cparams.nextn_layer_offset`, and the GLM4 / BailingMoE3
MTP graphs to copy. Nothing needs to be back-ported first.

> **Note on `draft-mtp-adaptive`:** that variant lives in a *different* checkout
> (`rwkv-related/adaptive-mtp/llama.cpp`) and is **not** in this fork. Ignore it
> here. It is a pure superset over the same `common_speculative_impl_draft_mtp`
> struct and can be merged later; it is not a dependency.

**Status of the GGUF.** `mtp_out/mtp_heads.gguf` (751 MB) already exists and has
been verified numerically — see §2.3. Treat it as a **frozen interface**.

---

## 1. What the model is

One MTP head from `train_temp/mtp/` (`MTPModule`, DeepSeek-V3-inspired,
MLP instead of a full transformer block). For hidden state `h ∈ R^C` and the
token embedding `tok ∈ R^C` of the *next* token:

```
x    = proj( concat( LN_h(h), LN_tok(tok) ) )      # 2C -> C
z    = x + mlp(x)                                  # mlp = LN, Linear(C->H), SiLU, Linear(H->C)
logits = W_head @ z                                # W_head is the FROZEN backbone lm-head
```

`C = 2560` (`n_embd`), `H = 5120` (`2*C`), `V = 65536`. Head 0 is 39,336,960
params = **78.7 MB in bf16**.

Five properties that the C++ must reproduce exactly. Getting any of them wrong
produces silently-wrong-but-plausible drafts, which is the worst failure mode
(it degrades acceptance rate rather than crashing):

1. **`LN_h`, `LN_tok`, `mlp[0]` are full `nn.LayerNorm`** — mean/variance, with
   **bias**, `eps = 1e-5`. Use **`LLM_NORM`**, *not* `LLM_NORM_RMS`. Every other
   MTP implementation in the tree (GLM4, BailingMoE3, Qwen3.5) uses RMSNorm and
   passes `nullptr` for bias. Do not copy that.
2. **Concat order is `[LN_h(h), LN_tok(tok)]` — hidden FIRST.** DeepSeek/GLM4 use
   `[enorm, hnorm]`. Both `proj` halves were trained in this order.
3. **No norm between `z` and the lm-head.** `logits = output @ z` directly. The
   GLM4 graph inserts `shared_head_norm` here; we must not.
4. **`h` fed in is the post-`output_norm` hidden**, i.e. exactly the tensor
   `src/models/rwkv7.cpp` already computes into `res->t_embd`. In the HF-side
   reference this is `ln_out(x)` and `head` is applied to it.
5. **Chaining**: the driver feeds the draft's *own* `t_h_nextn` back as `h` for
   the next draft depth, so the draft graph must publish `z` as `t_h_nextn`
   **as well as** consuming `h` as input. (This is true even for a single head —
   see §3.2.)

### 1.1 The recurrence identity (why the `h` round-trip is consistent)

```
target  t_h_nextn  = output_norm(backbone_hidden)        <- depth-0 input
draft   t_h_nextn  = z = MTPModule(h, emb(tok))          <- depth-(k+1) input
```

so `train`/`infer` agree at every depth without any extra plumbing.

---

## 2. The frozen GGUF interface

### 2.1 Metadata written by `scripts/convert_mtp_to_gguf.py`

| key | value | why |
|---|---|---|
| `general.architecture` | `"rwkv7"` | reuse the arch, add a trunk-less mode (see §4) |
| `rwkv7.block_count` | `K` (=1 now) | |
| `rwkv7.nextn_predict_layers` | `K` | ⇒ `n_layer() == block_count - nextn == 0` |
| `rwkv7.embedding_length` | `2560` | |
| `rwkv7.wkv.head_size` | `64` | read unconditionally by `load_arch_hparams` |
| `rwkv7.attention.decay_lora_rank` | `96` | ″ |
| `rwkv7.attention.iclr_lora_rank` | `96` | ″ |
| `rwkv7.attention.value_residual_mix_lora_rank` | `64` | ″ |
| `rwkv7.attention.gate_lora_rank` | `320` | ″ |
| `rwkv7.attention.layer_norm_epsilon` | `1e-5` | |
| `rwkv7.feed_forward_length` | `10240` | ″ |
| `rwkv7.attention.head_count` | `0` | required by llama.cpp, unused |
| `tokenizer.ggml.model` / `.tokens` / `.types` | rwkv / 65536 / … | the draft has its own sampler, so it needs a real vocab |

`block_count == nextn_predict_layers` is the load-bearing trick: `n_layer()`
becomes 0, so *every* block in the file is a NextN block, indexed `blk.0..blk.{K-1}`.
Two existing mechanisms then fire with no code change:

* `common_speculative_types_from_gguf()` (`common/speculative.cpp:2288`) probes
  for `blk.{block_count-1}.nextn.eh_proj.weight` and returns `DRAFT_MTP`.
  ⇒ **`--spec-type draft-mtp` is inferrable from the file alone.** The marker is
  present; do not rename `eh_proj`.
* `hparams.n_layer_nextn` is read generically from `{arch}.nextn_predict_layers`
  at `src/llama-model.cpp:1325`, and `llama_model_n_layer_nextn()` returns it.
  The driver then sets `n_mtp_layers = K` and `chain_heads = (K > 1)`.

### 2.2 Tensors

Shapes below are **ggml ne-order** (what `gguf_dump.py` prints):

```
blk.{k}.nextn.hnorm.weight      (C,)           F32   <- ln_h.weight
blk.{k}.nextn.hnorm.bias        (C,)           F32   <- ln_h.bias
blk.{k}.nextn.enorm.weight      (C,)           F32   <- ln_tok.weight
blk.{k}.nextn.enorm.bias        (C,)           F32   <- ln_tok.bias
blk.{k}.nextn.eh_proj.weight    (2C, C)        F16   <- proj.weight        [DETECT MARKER]
blk.{k}.nextn.mlp_norm.weight   (C,)           F32   <- mlp[0].weight
blk.{k}.nextn.mlp_norm.bias     (C,)           F32   <- mlp[0].bias
blk.{k}.nextn.mlp_up.weight     (C, H)         F16   <- mlp[1].weight
blk.{k}.nextn.mlp_down.weight   (H, C)         F16   <- mlp[3].weight

token_embd.weight               (C, V)         F16   duplicate of backbone emb
output.weight                   (C, V)         F16   duplicate of FROZEN backbone head
```

`mlp_norm` / `mlp_up` / `mlp_down` are **new** tensor names (nothing in
`LLM_TENSOR_NEXTN_*` covers an MLP). `hnorm.bias` / `enorm.bias` also don't
exist yet.

> **Shape gotcha.** `eh_proj` is stored `ne0=2C, ne1=C` — i.e. the *reverse* of
> the `{2*n_embd, n_embd}` you would naturally write in `create_tensor`. This is
> correct and consistent: gguf-py reverses the numpy shape into ne-order, so a
> PyTorch `(out, in)` Linear always lands as `ne0=in`. Just mirror the numbers in
> §2.2 and let the loader assert on them.

`token_embd`/`output` are duplicated on purpose (670 MB of the 751 MB file). The
head was trained against the frozen lm-head, and llama.cpp has no cross-model
weight-sharing path for a `-md` draft. A later optimisation could drop these two
and bind them to `ctx_tgt`'s tensors via `llama_get_ctx_other()`, but that is out
of scope — do not attempt it in the first PR.

### 2.3 Verification already performed

Performed against the real checkpoint (K=1, `step=2500`, `tokens=40.96M`):

| check | result |
|---|---|
| head forward, GGUF tensors (numpy) vs `MTPModule` (torch), same `h`/`tok` | max abs err **1.5e-05** |
| `output.weight` vs `.pth` `head.weight` | max abs err **2.98e-08** |
| `token_embd.weight` vs `.pth` `emb.weight` | max abs err **2.98e-08** |
| draft logits `output @ z` vs `head(z)`, argmax over 65536 | **exact match**, max abs err 1.3e-05 |
| `block_count - nextn_predict_layers` | `0` ✔ |
| auto-detect probe `blk.0.nextn.eh_proj.weight` | **FOUND** ✔ |
| tokenizer | 65536 tokens, `rwkv` ✔ |

Reproduce with `python scripts/verify_mtp_gguf.py` (in `train_temp/`; the
numeric round-trip block in it is the reference implementation of the head and
should be lifted into a C++ unit test — see §7).

---

## 3. How the driver uses the two contexts

Read `common_speculative_impl_draft_mtp` (`common/speculative.cpp:1328-1780`)
before writing anything. What it requires:

### 3.1 Of the target (`ctx_tgt`)

```cpp
n_embd = llama_model_n_embd_out(llama_get_model(ctx_dft));
GGML_ASSERT(n_embd == llama_model_n_embd_out(llama_get_model(ctx_tgt)));
n_mtp_layers = std::max(1, (int) llama_model_n_layer_nextn(llama_get_model(ctx_dft)));
llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
```

* `n_embd_out()` for RWKV-7 is `n_embd` = 2560 on both sides — assert satisfied.
* `masked == false` ⇒ the target must emit `t_h_nextn` **densely, for every token
  in the batch**, with no `inp_out_ids` filtering (see `get_embeddings_nextn_ith`,
  `src/llama-context.cpp:1010-1030`: the unmasked branch indexes by raw position).
* The driver then runs its own catch-up decode on `ctx_dft` by pairing each token
  with the *previous* position's target hidden (`pending_h` / `h_tgt + (k-1)*n_embd`).

### 3.2 Of the draft (`ctx_dft`)

* `llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true)` ⇒ **masked**:
  the draft graph must apply `build_inp_out_ids()` to `t_h_nextn`. The driver
  marks exactly the draft tokens with `batch.add(..., logits=true)`.
* Each draft step supplies **both** a token id and an embedding:
  `batch.add(id, pos, seq_id, true); batch.set_embd(idx, {h_row, n_embd})`.
  So the graph needs `build_inp_embd`-style `tokens` *and* an `h` input.
  `llm_graph_input_embd_h` (`src/llama-graph.h:144`) exists for exactly this;
  GLM4 uses it (`src/models/deepseek2.cpp:215-236`). BailingMoE3 instead stuffs
  `h` into `inp->embd` and names it `mtp_h_input` (`bailingmoe3.cpp:427-435`).
  Either works; `llm_graph_input_embd_h` is cleaner.
* The draft must publish `z` as `t_h_nextn` so the driver can chain depths
  (`h_row = llama_get_embeddings_nextn_ith(ctx_dft, i_last[seq_id])`).
* `ctx_dft` gets **recurrent** memory (`llm_arch_is_recurrent(LLM_ARCH_RWKV7)` is
  true — `src/llama-arch.cpp:1063`), with `n_layer() == 0` blocks. The MTP graph
  never calls `build_rs_inp`/`build_rwkv_token_shift_*`, so no state is touched.
  See §6-R1 for the residual risk.

---

### 3.3 `--spec-draft-n-max` — required to work

This is a **hard requirement**, not a nice-to-have. Nothing in the plan may
constrain it. The relevant facts, all verified in this fork:

* The flag sets `params.speculative.draft.n_max` (`common/arg.cpp:4136-4144`).
  It accepts any `value >= 0`; there is **no validation against the model**.
* **The default is `3`, not 1** — `common/common.h:327`,
  `int32_t n_max = 3; // maximum number of tokens to draft`.
* The MTP ctor applies exactly one cap, and only in the multi-head case
  (`common/speculative.cpp:1422-1432`):
  ```cpp
  chain_heads = n_mtp_layers > 1 && !is_mem_shared;
  if (chain_heads) {
      this->params.n_max = std::min(this->params.n_max, n_mtp_layers);   // :1425
  }
  this->n_max = this->params.n_max;
  ```
* The draft loop's only stop condition is
  `if (params.n_max <= (int) result.size())` (`:1698`), plus an optional
  `p_min` confidence cutoff (default `0.0` = disabled, `common/common.h:331`).

So two distinct regimes, and **both must work** for our arch:

| file | `chain_heads` | `--spec-draft-n-max N` behaviour |
|---|---|---|
| **K = 1** (what exists today) | `false` | **uncapped.** The single head is applied recursively N times: each step feeds its own `t_h_nextn` back as `h` and the token it just sampled as `tok`. |
| **K > 1** (converter supports it) | `true` | capped to `min(N, K)`; the K trained heads are walked via `llama_set_nextn_layer_offset` (`:1550`). |

### Arbitrary `n` on a single head is the *intended* mode, not a fallback

This is the single most important thing to get right, because it is easy to
"helpfully" add a guard that breaks it. **Modern MTP modules are expected to
serve any `n` even when the drafter was never trained for that `n`**, and
llama.cpp is built that way:

* The driver names the single-head path as a first-class mode, not a degraded
  one — `common/speculative.cpp:1340-1343`:
  ```cpp
  //   is_mem_shared (gemma4): shares the target KV, runs all heads in one graph.
  //   chain_heads (step35): n_mtp_layers trained heads, one per draft step.
  //   neither (qwen35 / qwen35moe): a single trained MTP head.
  ```
  Qwen3.5 ships **one** MTP layer and drafts `n_max = 3` by default — i.e.
  recursive reuse is the shipping default for a single-head MTP.
* `chain_heads` (one *trained* head per step, Step-3.5 style) is an
  **alternative specialisation**, not the prerequisite for deeper drafting.
  A K=1 file needs no K>1 support to draft n=8.
* Only *block* drafters clamp `n`: the docs say `--spec-draft-n-max` "is clamped
  to the draft model's trained block size" for `draft-dflash` and `draft-dspark`,
  and say nothing of the kind for `draft-mtp`. Their clamp exists because a block
  draft emits a fixed-size block in one graph; MTP has no such structure.

Consequence: our K=1 head **already** supports any `n`, because it is stateless
and pure — the recursion needs no KV, no state snapshot, no re-init. It is the
same functional form the chained training recurrence uses (`h_prev = z`), just
applied repeatedly.

Recursive reuse is also *exact*: speculative decoding accepts/rejects against the
target's distribution, so an out-of-distribution draft can only lower the
acceptance rate, never corrupt the output. The only open question is
quality-vs-cost, which is a tuning measurement (§7-T4b), not a design constraint.

**Consequences for the implementation:**

1. **Do not add `GGML_ASSERT(hparams.n_layer_nextn == 1)`.** `deepseek2.cpp:173`
   and `bailingmoe3.cpp:426` both have it; copying it would make a K>1 draft
   `abort()`. The §5.6 sketch is already K-generic (`il = n_layer() +
   nextn_layer_offset`); keep it that way and assert `>= 1` only.
2. **Do not add any cap, warning, or fallback tied to `n_max` for K=1** — no
   `min(n_max, n_mtp_layers)`, no "depth exceeds model capability" warning. The
   driver deliberately leaves K=1 uncapped.
3. `--spec-draft-n-max 0` does not cleanly disable drafting (the loop pushes one
   token before testing the limit), so document `1` as the practical floor.
4. Sizing: verifying `N` drafts needs `N + 1` tokens in one ubatch, so the
   target's `-b/--batch-size` must be `>= n_max + 1`, and `n_outputs_max` must
   cover `n_max + 1` per sequence (`common/speculative.cpp:2510-2513`). The
   target's unmasked `embd_nextn` buffer is `n_embd_out * n_batch`
   (`src/llama-context.cpp:2112-2125`). Nothing to write; just do not break it.
5. **Optional upstream nit (not required):** the K>1 cap at `:1425` is silent, so
   `--spec-draft-n-max 8` against a K=3 file quietly becomes 3. Raising that
   `SPC_TRC` to an `SPC_WRN` would help users. That is a
   `common/speculative.cpp` edit — do it only if you also want to own that file.

## 4. Design decision: extend `LLM_ARCH_RWKV7`, don't add an arch

**Chosen:** teach the existing `rwkv7` loader a *trunk-less* ("mtp_only") mode and
dispatch the MTP graph on `params.gtype == LLM_GRAPH_TYPE_DECODER_MTP`. This is
verbatim the GLM4-MTP pattern in `deepseek2.cpp:59-61` + `:163-168`, where a
`deepseek2` GGUF can be trunk-only, MTP-only, or both.

**Rejected:** a new `LLM_ARCH_RWKV7_MTP`. It buys nothing (the driver only cares
about `n_layer_nextn` and `n_embd_out`) and costs a full new arch registration,
vocab wiring, and a second copy of the RWKV-7 hparam parsing.

**Rejected:** putting the MTP block inside the *target* GGUF and loading it twice.
The user explicitly wants a separate `-md <path>`, and `ctx_tgt`/`ctx_dft` are
separate contexts anyway.

---

## 5. Change list

### 5.1 `src/llama-arch.h` — 5 new tensor enums

After `LLM_TENSOR_NEXTN_SHARED_HEAD_NORM` (currently `src/llama-arch.h:700`):

```cpp
    LLM_TENSOR_NEXTN_MLP_NORM,
    LLM_TENSOR_NEXTN_MLP_UP,
    LLM_TENSOR_NEXTN_MLP_DOWN,
```

The two bias tensors reuse `LLM_TENSOR_NEXTN_HNORM`/`_ENORM` with the `"bias"`
suffix, exactly as `output_norm`/`output_norm_b` do.

### 5.2 `src/llama-arch.cpp` — name templates + op classification

In the tensor-name table next to `src/llama-arch.cpp:584`:

```cpp
    { LLM_TENSOR_NEXTN_MLP_NORM,  "blk.%d.nextn.mlp_norm" },
    { LLM_TENSOR_NEXTN_MLP_UP,    "blk.%d.nextn.mlp_up"   },
    { LLM_TENSOR_NEXTN_MLP_DOWN,  "blk.%d.nextn.mlp_down" },
```

In the op table next to `src/llama-arch.cpp:974`:

```cpp
    {LLM_TENSOR_NEXTN_MLP_NORM,  {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_NEXTN_MLP_UP,    {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_NEXTN_MLP_DOWN,  {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
```

### 5.3 `src/llama-model.h` — extend `llama_layer_nextn`

`struct llama_layer_nextn` (`src/llama-model.h:224-235`) — add:

```cpp
    struct ggml_tensor * mlp_norm     = nullptr;   // weight
    struct ggml_tensor * mlp_norm_b   = nullptr;   // bias
    struct ggml_tensor * mlp_up       = nullptr;
    struct ggml_tensor * mlp_down     = nullptr;
    struct ggml_tensor * hnorm_b      = nullptr;   // ln_h bias
    struct ggml_tensor * enorm_b      = nullptr;   // ln_tok bias
```

Also add `bool is_mtp_only = false;` on `llama_hparams` if you want the graph to
assert on it (optional but recommended — see §5.4).

### 5.4 `src/models/rwkv7.cpp` — the loader (`load_arch_tensors`, line 46)

Currently every trunk tensor is created with flag `0` (required). Make them
conditional on a trunk existing:

```cpp
void llama_model_rwkv7::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;                    // gives n_layer = hparams.n_layer() (== 0 for a draft)

    // A trunk-less MTP draft has no blk.0.attn_norm. Same probe as GLM4-MTP.
    const bool mtp_only = hparams.n_layer_nextn > 0 && hparams.n_layer() == 0 &&
                          ml.get_weight("blk.0.attn_norm.weight") == nullptr;
    const int trunk_flags = mtp_only ? TENSOR_NOT_REQUIRED : 0;
    const int mtp_flags   = 0;

    tok_embd   = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, trunk_flags);
    tok_norm   = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), {n_embd}, trunk_flags);
    tok_norm_b = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "bias",   0), {n_embd}, trunk_flags);

    // NOTE: for a draft these MUST be TENSOR_NOT_REQUIRED. The head has no
    // output_norm (see §1 property 3) and `output` stays required because the
    // draft needs the frozen lm-head.
    output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, trunk_flags);
    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, trunk_flags);

    // The RWKV-7 filename for this is LLM_TENSOR_OUTPUT; TENSOR_DUPLICATED-free.
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, 0);

    // ... trunk loop: all create_tensor(...) calls take `trunk_flags` ...

    for (int i = 0; i < (int) hparams.n_layer_all; ++i) {
        if (i < (int) hparams.n_layer()) { /* trunk, flags = trunk_flags */ continue; }
        auto & layer = layers[i];
        const int f = mtp_flags;
        layer.nextn.eh_proj        = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,        "weight", i), {2 * n_embd, n_embd}, f);
        layer.nextn.hnorm          = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,         "weight", i), {n_embd}, f);
        layer.nextn.hnorm_b        = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,         "bias",   i), {n_embd}, f);
        layer.nextn.enorm          = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,         "weight", i), {n_embd}, f);
        layer.nextn.enorm_b        = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,         "bias",   i), {n_embd}, f);
        layer.nextn.mlp_norm       = create_tensor(tn(LLM_TENSOR_NEXTN_MLP_NORM,      "weight", i), {n_embd},  TENSOR_NOT_REQUIRED | f);
        layer.nextn.mlp_norm_b     = create_tensor(tn(LLM_TENSOR_NEXTN_MLP_NORM,      "bias",   i), {n_embd},  TENSOR_NOT_REQUIRED | f);
        layer.nextn.mlp_up         = create_tensor(tn(LLM_TENSOR_NEXTN_MLP_UP,        "weight", i), {n_embd, 2*n_embd}, TENSOR_NOT_REQUIRED | f);
        layer.nextn.mlp_down       = create_tensor(tn(LLM_TENSOR_NEXTN_MLP_DOWN,      "weight", i), {2*n_embd, n_embd}, TENSOR_NOT_REQUIRED | f);
    }
}
```

**Careful with the existing loop.** Today it reads `for (int i = 0; i < n_layer; ++i)`
`src/models/rwkv7.cpp:67`). `n_layer` from `LLAMA_LOAD_LOCALS` is
`hparams.n_layer()` (trunk only), so a draft would create *nothing*. Loop to
`n_layer_all` and split on `i < n_layer()`, exactly as `deepseek2.cpp:94-96` does.

`load_arch_hparams` (line 3) needs **no change**: it already reads every key we
write, and `n_layer_nextn` is loaded generically before arch dispatch.

### 5.5 `src/models/rwkv7.cpp` — graph dispatch (line 120)

```cpp
std::unique_ptr<llm_graph_context> llama_model_rwkv7::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP && hparams.n_layer_nextn > 0) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}
```

Declare `graph_mtp` next to `graph` in `src/models/models.h`
(`llama_model_rwkv7`'s nested structs).

### 5.6 `src/models/rwkv7.cpp` — `graph_mtp` (the actual work)

Model it on `bailingmoe3.cpp:424-563` (offset-aware, cleanest) + `deepseek2.cpp:170-410`
(norm/bias handling). Everything after `z` is plain matrix algebra — no WKV, no
token shift, no recurrent state, no positions.

```cpp
llama_model_rwkv7::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    GGML_ASSERT(hparams.n_layer_nextn > 0);

    const int il = hparams.n_layer() + cparams.nextn_layer_offset;
    GGML_ASSERT(cparams.nextn_layer_offset >= 0 &&
                cparams.nextn_layer_offset < (int) hparams.n_layer_nextn &&
                "nextn_layer_offset out of range");
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && layer.nextn.hnorm && layer.nextn.enorm);
    GGML_ASSERT(layer.nextn.hnorm_b && layer.nextn.enorm_b && "RWKV-7 MTP LayerNorms have bias");

    const int64_t n_embd = hparams.n_embd;
    const int64_t n_ff   = layer.nextn.mlp_up->ne[1];

    auto inp = std::make_unique<llm_graph_input_embd_h>(n_embd);
    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);
    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    // token side: the shared backbone embedding
    ggml_tensor * tok_embd = layer.nextn.embed_tokens
            ? ggml_get_rows(ctx0, layer.nextn.embed_tokens, inp->tokens)
            : ggml_get_rows(ctx0, model.tok_embd,            inp->tokens);

    res->add_input(std::move(inp));

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // 1. LayerNorm (mean/var + bias) -- NOT LLM_NORM_RMS
    ggml_tensor * h_norm = build_norm(inp->h, layer.nextn.hnorm, layer.nextn.hnorm_b, LLM_NORM, il);
    cb(h_norm, "mtp_hnorm", il);
    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, layer.nextn.enorm_b, LLM_NORM, il);
    cb(e_norm, "mtp_enorm", il);

    // 2. concat(HIDDEN, TOKEN) -- hidden first. Do not reorder.
    ggml_tensor * cur = ggml_mul_mat(ctx0, layer.nextn.eh_proj,
                                     ggml_concat(ctx0, h_norm, e_norm, 0));
    cb(cur, "mtp_eh_proj", il);

    // 3. residual MLP
    if (layer.nextn.mlp_up) {
        ggml_tensor * res_in = cur;
        ggml_tensor * m = build_norm(cur, layer.nextn.mlp_norm, layer.nextn.mlp_norm_b, LLM_NORM, il);
        cb(m, "mtp_mlp_norm", il);
        m = ggml_silu(ctx0, ggml_mul_mat(ctx0, layer.nextn.mlp_up, m));
        cb(m, "mtp_mlp_act", il);
        m = ggml_mul_mat(ctx0, layer.nextn.mlp_down, m);
        cur = ggml_add(ctx0, res_in, m);
        cb(cur, "mtp_mlp_out", il);
    }

    // 4. publish z as the next depth's h (masked -> only rows the driver flagged)
    ggml_tensor * h_nextn = cur;
    if (inp_out_ids) {
        h_nextn = ggml_get_rows(ctx0, h_nextn, inp_out_ids);
    }
    cb(h_nextn, "h_nextn", -1);
    res->t_h_nextn = h_nextn;

    // 5. frozen lm-head, applied DIRECTLY to z (no shared_head_norm)
    GGML_ASSERT(model.output && "RWKV-7 MTP draft needs output.weight");
    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
```

Ordering note: `res->t_h_nextn` is set **before** the `inp_out_ids` gather in
deepseek2 but **after** in bailingmoe3. We need the *gathered* version, because
the driver's `get_embeddings_nextn_ith` in masked mode resolves rows through
`output_resolve_row(i)`.

### 5.7 `src/models/rwkv7.cpp` — target side, ~3 lines (line ~200)

The trunk graph must expose the post-`output_norm` hidden. Today:

```cpp
    cur = build_norm(cur, model.output_norm, model.output_norm_b, LLM_NORM, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;
    cur = build_lora_mm(model.output, cur, model.output_s);
```

becomes (mirroring `deepseek2.cpp:694-704` exactly):

```cpp
    cur = build_norm(cur, model.output_norm, model.output_norm_b, LLM_NORM, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;                    // <-- the h the MTP head was trained on

    if (cparams.embeddings_nextn && !cparams.embeddings_nextn_masked && inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;
    cur = build_lora_mm(model.output, cur, model.output_s);
```

**This is the whole target-side change, and it is inert when speculative decoding
is off**: `llama_context` only reads `t_h_nextn` when `cparams.embeddings_nextn`
is set (`src/llama-context.cpp:1548`, `:1923`), and only allocates `embd_nextn`
then (`:2112`). The one unavoidable cost is `ggml_set_output(t_h_nextn)` in
`llm_graph_result::set_outputs` (`src/llama-graph.cpp:1373`), which pins an
extra `[n_embd, n_tokens]` f32 buffer ≈ 10 KB/token. Do **not** try to gate it on
`cparams`; deepseek2 does not, and the graph is re-reserved on param change anyway
(`llm_graph_params::can_reuse`, `src/llama-graph.h:873`).

Sanity-check that `n_layer() == 0` doesn't break `build_rs_inp` for the *target*
— it doesn't, the target has 32 trunk layers.

### 5.8 No changes needed

* `common/speculative.cpp`, `common/speculative.h`, `common/arg.cpp` — the driver
  is generic over `n_layer_nextn`; `-md` + `--spec-type draft-mtp` already parse.
* `src/llama-model.cpp` — memory creation for `LLM_ARCH_RWKV7` takes the
  `llm_arch_is_recurrent` branch (`:2614`) and ignores `ctx_type`; see §6-R1.
* `gguf-py/gguf/constants.py` — the converter writes literal tensor names and
  never consults `MODEL_TENSORS`. (Add the three names anyway if you want
  `gguf_dump.py`/quantization tooling to classify them nicely.)

---

## 6. Risks, ranked

**R1 — recurrent memory with `n_layer() == 0` (highest risk).**
`llama_memory_recurrent` is constructed for the draft (`src/llama-model.cpp:2614`)
and will size its state from `hparams.n_embd_s()` × the layer set. An RWKV-7
`n_embd_s()` is the WKV state (`head_size² × n_head` per layer), so if the layer
filter defaults to "all `n_layer_all` layers" the draft may allocate one full WKV
state (~2560×64 f32 ≈ 0.65 MB) it never uses. Probably harmless, but:
* verify `llama-cli` startup logs for the draft context and confirm no absurd
  allocation;
* if it does allocate, the clean fix is to give the RWKV-7 MTP context a
  `layer_filter` that excludes nextn layers — copy the `mtp_on_hybrid_qwen`
  handling at `src/llama-model.cpp:2606` / `:2743-2756`.
Do **not** hack `llm_arch_is_recurrent` to return false for RWKV7; that breaks
the trunk.

**R2 — accepting a wrong-but-plausible head.** An RMSNorm-vs-LayerNorm mixup, a
swapped concat order, or an extra `shared_head_norm` will still *run* and still
produce *tokens*; it just quietly halves the acceptance rate. §7-T2 exists solely
to catch this class of bug. Do not skip it.

**R3 — `inp_out_ids` on the wrong side of `t_h_nextn`.** Target must be *unmasked*
(dense rows); draft must be *masked* (gathered rows). Getting it backwards gives
an off-by-one hidden state and garbage drafts, or an out-of-range read.

**R4 — no MLP.** The converter tolerates an MLP-free head (`--use-mlp False` in
training). `mlp_up` is `TENSOR_NOT_REQUIRED`; the `if (layer.nextn.mlp_up)` guard
handles it. Also assert `mlp_norm`/`mlp_down` exist iff `mlp_up` does.

**R5 — K > 1 chaining (a correctness trap, not a feature gate).** `bailingmoe3.cpp:426`
and `deepseek2.cpp:173` both `GGML_ASSERT(n_layer_nextn == 1)`; **do not copy that
assert** — see §3.3. Our converter emits K heads (`block_count = nextn = K`) and
the driver sets `chain_heads = true` for K > 1 and walks them with
`llama_set_nextn_layer_offset`. The only thing the graph must do is use
`il = hparams.n_layer() + cparams.nextn_layer_offset`, which §5.6 already does.

K > 1 is *not* needed for deep drafting — a single head serves any `n` (§3.3).
Chain heads are a Step-3.5-style specialisation (one trained head per step).

Beware one asymmetry: with `chain_heads`, `process()` re-runs the whole verify
batch **once per head** (`common/speculative.cpp:1550`) and calls
`llama_memory_seq_rm` between heads, because attention-based MTP heads each own a
KV layer. Our head is stateless, so that loop is redundant compute over the
prefix — correct, just wasteful. Not worth special-casing unless the K>1 profile
looks bad.

**R6 — `token_shift_count`.** Defaults to `2` in `llama_hparams`, and the trunk
graph asserts `== 2` (`src/models/rwkv7.cpp:126`). The MTP graph must not inherit
that assert (it has no token shift). If you copy the trunk graph preamble, delete
the assert.

---

## 7. Test plan

**T1 — loader.** `llama-cli -m <draft>.gguf -md <draft>.gguf ...` or simply load
the draft as a target to confirm it parses: expect `n_layer_all = 1`,
`n_layer_nextn = 1`, `n_layer = 0`, all 11 tensors bound, no missing-required
errors. Then load it as `-md` and confirm the log shows
`adding speculative implementation 'draft-mtp'` with `n_embd=2560`.

**T2 — numeric parity (must-do).** Add `tests/test-rwkv7-mtp.cpp` (pattern:
`tests/test-arg-parser.cpp` for registration, `tests/CMakeLists.txt` for the
target — this tree has 52 test binaries already). Feed a fixed `h` and token id,
evaluate the graph, and compare `z` (= `t_h_nextn`) and `logits` against hardcoded
reference values from the Python round-trip. Practically: dump `h` and the
expected `z`/`logits` from `scripts/verify_mtp_gguf.py` into a binary/JSON fixture, check
in the fixture, and assert `max|Δ| < 1e-3` (f16 weights). **This is the test that
catches R2.**

**T3 — argmax agreement with PyTorch.** For ~100 random (h, token) pairs, the
GGUF-side argmax over 65536 logits must match the `mtp` package's. Already verified in
Python for 5 positions; extend to 100+ in C++.

**T4 — acceptance rate, end to end.** Build the target RWKV-7 GGUF, then:

```bash
llama-cli -m rwkv7-2.9b.gguf -md mtp_heads.gguf --spec-type draft-mtp \
          -ngl 99 -c 4096 -n 256 -p "Once upon a time"
```

**Acceptance criterion: mean accepted length ≈ 2.6 tokens/step**, i.e. draft
acceptance ≈ 0.63 for the tail token. This is the number measured offline by
`scripts/eval_mtp.py` on the held-out split:

```
 depth  tokens ahead  acceptance      n
  main             1      0.8687   65,464
 mtp_1             2      0.6299   65,432
```

If you land anywhere in **0.55–0.75** you are correct; R2 mixes up show up as
≈0.2–0.35. Note the offline number is greedy-ish and the driver adds a
`top_k=10` draft sampler with a `p_min` cutoff, so expect a little drift.

**T4b — `--spec-draft-n-max` sweep (required).** With the K=1 draft, run
`--spec-draft-n-max` in `{1, 2, 3, 4, 8}` and record tokens/s and mean accepted
length for each. This is a **characterisation**, not a hunt for a bug: every
value is a supported configuration (§3.3) and the default is `3`.

What to confirm:

* every value is accepted — no clamp, no warning, no abort. This is the headline
  requirement;
* acceptance is non-increasing in draft depth, and tokens/s is unimodal (some
  depth washes out before rejection cost dominates). Report the argmax; do not
  assume it is 1 or `n_max`;
* `n = 1` vs the baseline (no `-md`) isolates draft overhead from draft benefit;
* with a **K=3** draft (`mtp_out_proto/mtp_heads.pt` run through
  `scripts/convert_mtp_to_gguf.py`), `--spec-draft-n-max 8` runs without asserting and
  behaves as `min(8, 3)`. This is the regression test for the
  `n_layer_nextn == 1` assert trap in §3.3-1, and the only thing that exercises
  `chain_heads` at all;
* `-b >= n_max + 1` on every run (§3.3-4).

**T5 — no regression when unused.** Run the trunk without `-md`/`--spec-type` and
confirm identical logits/baseline throughput modulo the ~10 KB/token `t_h_nextn`
buffer.

**T6 — multi-seq.** `llama-server` with `-np 2` and two concurrent requests
(`--spec-type draft-mtp`), because the driver batches sequences (`n_seq`, `i_batch_beg`/
`i_batch_end`) and `pending_h` is per-sequence. Single-stream `llama-cli` will not
exercise this.

---

## 8. Reference index (all verified)

**The driver — read this first**
* `common/speculative.cpp:1328-1780` — `common_speculative_impl_draft_mtp`.
  `:1328` struct, `:1364-1442` ctor (asserts, `chain_heads`), `:1444` dtor,
  `:1482-1595` `process()` (catch-up decode + `verify_h`), `:1600-1745` `draft()`
  (the sampling loop), `:1751-1780` `accept()` (`pending_h = verify_h[n_accepted]`).
* `common/speculative.cpp:2288-2323` — `common_speculative_types_from_gguf`;
  the `blk.{block_count-1}.nextn.eh_proj.weight` probe.
* `common/speculative.cpp:2662` — where the impl is registered.
* `src/llama-ext.h:96-108` — `llama_set_embeddings_nextn`,
  `llama_get_embeddings_nextn[_ith]`, `llama_set_nextn_layer_offset`,
  `llama_model_n_layer_nextn`.
* `src/llama-context.cpp:1010-1030` — masked vs unmasked row resolution.
* `src/llama-context.cpp:1246` — `set_nextn_layer_offset`.
* `src/llama-context.cpp:1548`, `:1923`, `:2112` — `embd_nextn` allocation gating.

**Templates to copy**
* `src/models/deepseek2.cpp:59-61` — `mtp_only` / `trunk_only` probes.
* `src/models/deepseek2.cpp:94-96` — `n_layer_all` loop + `trunk_flags`/`mtp_flags`.
  (This is the exact shape §5.4 needs; the RWKV-7 loop at
  `src/models/rwkv7.cpp:67` must be widened to match it.)
* `src/models/deepseek2.cpp:163-168` — `build_arch_graph` dispatch on
  `LLM_GRAPH_TYPE_DECODER_MTP`.
* `src/models/deepseek2.cpp:170-410` — full `graph_mtp` (uses
  `llm_graph_input_embd_h`).
* `src/models/deepseek2.cpp:392-393`, `:694-704` — the MTP-side and **target-side**
  `h_nextn` blocks. The target-side one is §5.7.
* `src/models/bailingmoe3.cpp:424-563` — the `nextn_layer_offset` variant, and the
  one that uses `llm_graph_input_embd` + a `"mtp_h_input"` name instead.
  Its tail (`:555-563`) is the **masked-gather-then-frozen-head** pattern this
  graph needs verbatim:
  ```cpp
  cb(cur, "h_nextn", -1);
  res->t_h_nextn = cur;
  cur = ggml_get_rows(ctx0, cur, inp_out_ids);
  cur = ggml_mul_mat(ctx0, model.output, cur);
  ```
* `src/llama-graph.h:144` — `llm_graph_input_embd_h` (`tokens` + `h`).
* `src/llama-model.cpp:1323`, `:1325` — `block_count` / `nextn_predict_layers`.
* `src/llama-model.cpp:2614` — recurrent memory for RWKV-7 (and R1).

**Files to edit**
| file | what |
|---|---|
| `src/llama-arch.h` | +3 tensor enums |
| `src/llama-arch.cpp` | +3 name templates, +3 op classifications |
| `src/llama-model.h` | extend `llama_layer_nextn` |
| `src/models/rwkv7.cpp` | loader flags, graph dispatch, `graph_mtp`, 3-line target change |
| `src/models/models.h` | declare `llama_model_rwkv7::graph_mtp` |
| `tests/CMakeLists.txt` + new `tests/test-rwkv7-mtp.cpp` | T2/T3 |

**RWKV-7 tensor names for reference** (`src/llama-arch.cpp`, the
`blk.%d.time_mix_*` / `blk.%d.channel_mix_*` group):
`blk.%d.attn_norm[._2]`, `token_embd`, `token_embd_norm`, `output`, `output_norm`.
`LLM_KV_TOKEN_SHIFT_COUNT` = `"%s.token_shift_count"`; RWKV-7's graph asserts it
is `2`.

---

## 9. Local artefacts

| path | what |
|---|---|
| `train_temp/scripts/convert_mtp_to_gguf.py` | the converter (done, verified) |
| `train_temp/mtp_out/mtp_heads.gguf` | K=1 draft, 751 MB, verified |
| `train_temp/scripts/verify_mtp_gguf.py` | the numeric round-trip / fixture generator |
| `train_temp/mtp/` | `MTPModule` — the reference implementation |
| `train_temp/MTP_README.md` | training-side docs |
| `train_temp/data/val.bin`, `scripts/eval_mtp.py` | offline acceptance measurement |
| `train_temp/mtp_out_proto/mtp_heads.pt` | K=3 prototype (R5 follow-up) |

Build the target GGUF with llama.cpp's own `convert_hf_to_gguf.py` +
`conversion/rwkv.py` (`Rwkv7ForCausalLM`); the draft GGUF's vocab is byte-identical
to it because `scripts/convert_mtp_to_gguf.py` reimplements `_set_vocab_rwkv_world`
(`conversion/base.py:2031-2075`) from the same `rwkv_vocab_v20230424.txt`.
