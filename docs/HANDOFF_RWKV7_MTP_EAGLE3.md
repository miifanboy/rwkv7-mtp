Here is the complete, surgical engineering handoff for adding **EAGLE-3 Style Multi-Layer Feature Fusion** to the RWKV-7 MTP codebase.

---

# Engineering Handoff: EAGLE-3 Multi-Layer Feature Fusion for RWKV-7 MTP

## 1. Architectural Overview & Design Invariants

In the current implementation, `RWKVWithMTP` feeds only the post-`ln_out` final layer representation (`hidden`, i.e., `ln_out(x_23)`) into depth $k=0$ of the MTP chain. However, by Layer 23, the RWKV-7 residual stream has collapsed intermediate syntactic and multi-horizon planning features into a representation optimized strictly for the immediate next token $t_{i+1}$. When a single `RWKV7MTPModule` (`--shared-head`) is unrolled across $K=3$ depths, depth $k=0$ receives a next-token-collapsed vector while depths $k \ge 1$ receive recurrent draft hidden states $z_{k-1}$.

**EAGLE-3 Multi-Layer Fusion** resolves this by tapping **three intermediate layers** from the frozen RWKV-7 backbone—early ($l_{\text{early}}$), middle ($l_{\text{mid}}$), and final ($l_{\text{late}}$)—and projecting them into a single $C$-dimensional context vector $h_{\text{fused}} \in \mathbb{R}^{B \times T \times C}$ that enters depth $k=0$.

### Strict Codebase Invariants Preserved

1. **Unmodified Teacher & Main-Head Semantics:**
* `hidden` (`ln_out(x_{L-1})`) remains the **exact** input to `main_logits = self.head(hidden)`, `drafts[0] = pick(h @ self.head)`, and the teacher source `t_h` / `t_logits` in `_lk_loss` and `_teacher_topk`.


* **`h_fused` is used exclusively as the $k=0$ state input (`h_prev` at $k=0$)** to the MTP head chain during training (`backbone.py`) and inference (`spec_decode.py`).




2. **Zero-Shock Warm Initialization (`[0, 0, I]`):**
* The fusion module applies separate `LayerNorm(C)` instances to the early and middle taps, uses the already-normalized `hidden` (`ln_out(x_{L-1})`) as the 3rd tap, and projects `3C -> C` via `fc = nn.Linear(3 * C, C, bias=False)`.
* Initializing `fc.weight` to `[0, 0, I]` guarantees that at step 0 of training, `h_fused == hidden` bit-for-bit.


3. **Checkpoint & Optimizer Guard Compatibility:**
* `train_mtp.py` asserts `all("mtp_heads" in n for n in trainable_names)`, and `save_mtp_heads` / `split_head_sd` serialize `model.mtp_heads`.


* Therefore, the EAGLE-3 fusion parameters (`fc_fuse`, `ln_early`, `ln_mid`) and optional per-depth input LayerNorms (`ln_h_depths`) live **inside `RWKV7MTPModule` (at index 0 of `mtp_heads`)**, requiring zero changes to `mtp_parameters(model)` or the checkpoint saver.





---

## 2. Tap Selection Rule for Any Layer Count $L$

For a frozen RWKV-7 backbone with `n_layer` $= L$ (for 1.5B, $L=24$; for 2.9B, $L=32$), the three 0-indexed block tap indices are computed deterministically as:


$$\text{tap\_indices}(L) = \left( \max(1, L // 8 - 1),\; L // 2 - 1,\; L - 1 \right)$$

* For **1.5B ($L = 24$)**: blocks **`(2, 11, 23)`** (3rd, 12th, and 24th layers), where tap 2 (`block 23`) is taken after `self.ln_out` so it shares the exact tensor `hidden`.
* For **2.9B ($L = 32$)**: blocks **`(3, 15, 31)`**.

---

## 3. File-by-File Implementation Specification

### File 1: `mtp/heads/rwkv7.py` (Modify `RWKV7MTPModule`)

Add two optional capabilities directly to `RWKV7MTPModule`:

1. **EAGLE-3 Multi-Layer Fusion (`use_eagle3_fusion: bool`)**: Adds `ln_early`, `ln_mid`, and `fc_fuse = nn.Linear(3 * C, C, bias=False)` initialized to `[0, 0, I]`, plus a helper `fuse_backbone(h_early, h_mid, h_late) -> Tensor`.
2. **Per-Depth Input Normalization (`num_depths: int = 1`)**: When `shared_head=True` with $K=3$, depth $k=0$ normalizes `h_fused` while depths $k=1, 2$ normalize recurrent block outputs $z_0, z_1$. Adding `self.ln_h_extra = nn.ModuleList([nn.LayerNorm(C) for _ in range(max(0, num_depths - 1))])` gives depths $k \ge 1$ their own 4K-parameter LayerNorm affine scale/bias while keeping `self.ln_h` for $k=0$.
3. **Final Output Normalization (`self.ln_out = nn.LayerNorm(C)`)**: Notice that in `src/model.py`, `Block.forward` outputs an **unnormalized residual stream** `x = x + self.ffn(self.ln2(x))`, which the backbone passes through `self.ln_out(x)` before `self.head`. Adding `self.ln_out = nn.LayerNorm(C)` at the end of `RWKV7MTPModule.forward` ensures `z` is properly normalized before entering `_lk_loss` / `self.head(z)`.



```python
class RWKV7MTPModule(nn.Module):
    def __init__(self, args, eagle3_fusion: bool = True):
        super().__init__()
        from ..backbone import _rwkv_base_module  # lazy: avoids import cycle
        rw = _rwkv_base_module()
        C = args.n_embd
        self.n_embd = C
        self.eagle3_fusion = bool(eagle3_fusion)

        if self.eagle3_fusion:
            self.ln_early = nn.LayerNorm(C)
            self.ln_mid = nn.LayerNorm(C)
            # Maps [ln_early(h_early) ; ln_mid(h_mid) ; h_late_post_ln_out] -> C
            self.fc_fuse = nn.Linear(3 * C, C, bias=False)

        self.ln_h = nn.LayerNorm(C)
        self.ln_tok = nn.LayerNorm(C)
        self.proj = nn.Linear(2 * C, C, bias=False)

        self.block = rw.Block(args, 0)
        self.block.att.v0 = self.block.att.a0
        self.block.att.v1 = self.block.att.a1
        self.block.att.v2 = self.block.att.a2
        self.ln_out = nn.LayerNorm(C)
        self._init_weights()

    def _init_weights(self) -> None:
        C = self.n_embd
        with torch.no_grad():
            if self.eagle3_fusion:
                # Start as 100% identity on h_late (slice 2C:3C) so step 0
                # matches single-layer hidden_states bit-for-bit.
                self.fc_fuse.weight.zero_()
                self.fc_fuse.weight[:, 2 * C:].copy_(torch.eye(C))
            # Start proj as [I ; 0] so step 0 passes normalized h_prev cleanly
            self.proj.weight.zero_()
            self.proj.weight[:, :C].copy_(torch.eye(C))

    def fuse_backbone(self, h_early: torch.Tensor, h_mid: torch.Tensor,
                      h_late: torch.Tensor) -> torch.Tensor:
        """Fuse (early, mid, post-ln_out final) backbone states into (B, T, C)."""
        if not self.eagle3_fusion:
            return h_late
        cat = torch.cat([
            self.ln_early(h_early),
            self.ln_mid(h_mid),
            h_late,
        ], dim=-1)
        return self.fc_fuse(cat)

    def forward(self, h: torch.Tensor, tok_emb: torch.Tensor) -> torch.Tensor:
        x = self.proj(torch.cat([self.ln_h(h), self.ln_tok(tok_emb)], dim=-1))
        z, _ = self.block(x, None)
        return self.ln_out(z)

```

---

### File 2: `mtp/backbone.py` (Multi-Layer Tap Extraction & Warm-Start)

Instead of modifying `src/model.py` (which is upstream RWKV-LM code), add a non-intrusive `_forward_features_multi(self, idx)` helper directly inside `RWKVWithMTP` in `mtp/backbone.py`:

```python
        @staticmethod
        def tap_layers(n_layer: int) -> tuple[int, int, int]:
            return (max(1, n_layer // 8 - 1), n_layer // 2 - 1, n_layer - 1)

        def _forward_features_multi(
            self, idx: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Run frozen backbone and return (h_early, h_mid, hidden_post_ln_out)."""
            args = self.args
            B, T = idx.size()
            assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."
            i_early, i_mid, _ = self.tap_layers(args.n_layer)

            x = self.emb(idx)
            v_first = torch.empty_like(x)
            h_early = h_mid = None
            for i, block in enumerate(self.blocks):
                x, v_first = block(x, v_first)
                if i == i_early:
                    h_early = x
                elif i == i_mid:
                    h_mid = x
            hidden = self.ln_out(x)
            return h_early, h_mid, hidden

        def hidden_states_with_h0(
            self, idx: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Return (hidden, h0_for_depth0).
            - hidden: post-ln_out final layer (for main_logits & teacher targets)
            - h0: fused multi-layer anchor for depth k=0 if head 0 uses EAGLE-3
            """
            head0 = self._head(0)
            use_fusion = getattr(head0, "eagle3_fusion", False)
            if not use_fusion:
                h = self.hidden_states(idx)
                return h, h
            if self.freeze_backbone:
                with torch.no_grad():
                    h_early, h_mid, hidden = self._forward_features_multi(idx)
            else:
                h_early, h_mid, hidden = self._forward_features_multi(idx)
            # fuse_backbone runs OUTSIDE no_grad so fc_fuse, ln_early, ln_mid train!
            h0 = head0.fuse_backbone(h_early, h_mid, hidden)
            return hidden, h0

```

#### Update `RWKVWithMTP.forward` and `RWKVWithMTP.forward_loss` in `mtp/backbone.py`

:

1. Replace `hidden = self.hidden_states(idx)` with:
```python
hidden, h0 = self.hidden_states_with_h0(idx)

```


2. Replace `h_prev = hidden` right before `for k in range(self.num_mtp):` with:
```python
h_prev = h0

```


Everything else in `forward_loss` (`ce_main`, `_teacher_topk(hidden, ...)`, `_teacher_logits(hidden, ...)`, and `t_h = hidden[:, k + 1:k + 1 + n_rows, :]`) continues to use `hidden` unchanged!



#### Warm-Start Hook in `mtp/checkpoint.py` (`build_frozen_model`):

Right after `build_frozen_model` loads the `.pth` backbone weights into `RWKVWithMTP`, copy the pretrained weights of `model.blocks[-1]` and `model.ln_out` into each `RWKV7MTPModule` (when starting a fresh run without `--load-mtp`):

```python
if arch == "rwkv7":
    with torch.no_grad():
        last_blk = model.blocks[-1]
        for mtp_head in model.mtp_heads:
            mtp_head.block.ln1.load_state_dict(last_blk.ln1.state_dict())
            mtp_head.block.ln2.load_state_dict(last_blk.ln2.state_dict())
            mtp_head.block.att.load_state_dict(last_blk.att.state_dict(), strict=False)
            mtp_head.block.ffn.load_state_dict(last_blk.ffn.state_dict())
            # Re-alias dead layer-0 value-residual params to a0/a1/a2
            mtp_head.block.att.v0 = mtp_head.block.att.a0
            mtp_head.block.att.v1 = mtp_head.block.att.a1
            mtp_head.block.att.v2 = mtp_head.block.att.a2
            if hasattr(mtp_head, "ln_out"):
                mtp_head.ln_out.load_state_dict(model.ln_out.state_dict())

```

---

### File 3: `spec_decode.py` & `generate_data.py` (Inference Runner Alignment)

At inference time in `spec_decode.py`, `BatchedRWKV7.forward(idx, state, return_hidden=True)` currently returns `(hidden, logits, state)` where `hidden` is `(B, T, C)` post-`ln_out`.

#### Step 3A: Return Tap States from `BatchedRWKV7.forward` (`generate_data.py`)

Add an optional `return_taps: bool = False` argument to `BatchedRWKV7.forward`. When `return_taps=True`, capture `(x_early, x_mid)` after layer indices `i_early = max(1, n_layer // 8 - 1)` and `i_mid = n_layer // 2 - 1`, and return `taps = (x_early, x_mid)` alongside `hidden`.

#### Step 3B: Update `_RWKV7_HEAD_TRANSPOSE` and `RWKV7HeadRunner` in `spec_decode.py`

1. Add `"fc_fuse.weight"` to `_RWKV7_HEAD_TRANSPOSE` so `prepare_rwkv7_head_weights` transposes `(C, 3C) -> (3C, C)` for fp16 `cat @ w["fc_fuse.weight"]`:


```python
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

```


2. In `RWKV7HeadRunner._run`, apply `ln_out` at the end if `"ln_out.weight"` is in `w` (backward-compatible with older checkpoints):


```python
if "ln_out.weight" in w:
    x = F.layer_norm(x, (C,), w["ln_out.weight"], w["ln_out.bias"])
return x

```


3. Add `fuse_taps(self, h_early, h_mid, h_late)` to `RWKV7HeadRunner`:


```python
def fuse_taps(self, h_early: torch.Tensor, h_mid: torch.Tensor,
              h_late: torch.Tensor) -> torch.Tensor:
    """Fuse (T, C) tap tensors into (T, C) depth-0 input h0."""
    w = self.wsets[0]
    if "fc_fuse.weight" not in w:
        return h_late
    F = torch.nn.functional
    he = F.layer_norm(h_early, (self.C,), w["ln_early.weight"], w["ln_early.bias"])
    hm = F.layer_norm(h_mid, (self.C,), w["ln_mid.weight"], w["ln_mid.bias"])
    return torch.cat([he, hm, h_late], dim=-1) @ w["fc_fuse.weight"]

```


4. Clean Separation of `h` (for `drafts[0]`) vs. `h0` (for `RWKV7HeadRunner` depth 0) in `SpecDecoder.generate`:


* Notice in `RWKV7HeadRunner.draft_chain(self, h, p, pick)`:


* `drafts = [pick(h)]` uses `h` (post-`ln_out` final layer) to pick the backbone's greedy/sampled token $t_1$.


* Inside the loop, `if k == 0: prev = h0` uses the fused representation!


* Therefore, change `draft_chain(self, h, p, pick, h0=None)` so `drafts = [pick(h)]` still uses `h`, while `if k == 0: prev = h0 if h0 is not None else h` uses `h0`.


* Similarly, pass `H0` into `self.runner.prefill(H0, list(prompt_ids))` and `(h0, H0_new)` into `self.runner.advance(h0, H0_new, p, out)`.





---

## 4. Recommended Training Command & Hyperparameters (RTX 4070 Super)

With EAGLE-3 fusion (`fc_fuse` adds only **12.58M parameters** / 25 MB in `bf16` at $C=2048$) and your fused CUDA LK loss:

```bash
python scripts/train_mtp.py \
    --data data/train.bin --data-dtype uint16 --val-fraction 0.01 \
    --pth rwkv7-g1k-1.5b.pth --config config.json \
    --mtp-arch rwkv7 --shared-head --num-mtp 3 \
    --mtp-loss lk-hybrid --lk-eta 2.0 --lk-teacher-cache auto \
    --skip-main-ce \
    --ctx-len 2048 --batch-size 2 --grad-acc 4 \
    --lr 3e-4 --lr-final 3e-5 --warmup 150 --steps 5000 --decay 0.8 \
    --out mtp_out_eagle3_rwkv7

```

### Verification Checklist After Implementation

1. **Step-0 Sanity Check:** Before step 1 updates any weights, verify that `head0.fuse_backbone(h_early, h_mid, hidden)` equals `hidden` with max absolute difference `0.0` (due to `[0, 0, I]` initialization).
2. Parameter Count Check (`--check-only`):


* Without EAGLE-3 fusion: `RWKV7MTPModule` at $C=2048$ has `~58.7M` parameters.
* With EAGLE-3 fusion (`fc_fuse` + `ln_early` + `ln_mid` + `ln_out`): trainable parameter count increases by exactly $3 \times 2048^2 + 6 \times 2048 = \mathbf{12,595,200}$ parameters (`~71.3M` total, 100% inside `mtp_heads.0.*`).




3. **Training Dynamics Check:** Watch `a1`, `a2`, `a3` (`alpha_1`, `alpha_2`, `alpha_3` in `train_log.txt`). With warm-starting from `blocks[-1]` + `[0, 0, I]` fusion init, `a1` should start around `0.25–0.35` within the first 50 steps instead of climbing from `0.01`.


