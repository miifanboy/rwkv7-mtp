"""
Full RWKV-7-block MTP head.

Unlike :class:`~mtp.heads.mlp.MTPModule`, this head mixes across stream
positions and therefore carries a recurrent WKV state at inference.  That extra
state is exactly where its higher acceptance comes from.

EAGLE-3 multi-layer feature fusion (optional, default on): the residual stream
of a deep RWKV-7 collapses intermediate syntactic / multi-horizon features by
the last layer, and depth k=0 of the draft chain otherwise receives only the
post-``ln_out`` next-token-collapsed vector.  When ``eagle3_fusion`` is set the
head additionally owns

* ``ln_early`` / ``ln_mid`` -- separate LayerNorms for the early and middle
  backbone taps (raw residual-stream states taken mid-backbone),
* ``fc_fuse`` -- a ``3C -> C`` projection initialized to ``[0, 0, I]``, so at
  step 0 of training ``fuse_backbone(h_early, h_mid, h_late) == h_late``
  bit-for-bit (zero-shock warm start; training then learns how much of the
  early/middle features to let in).

``h_late`` is the already-normalized post-``ln_out`` backbone hidden -- the
exact tensor that feeds the main head -- so the fusion is a strict *addition*
to depth k=0's state input; the main head and the teacher targets are
untouched.  The fusion parameters live inside the head module, i.e. under
``mtp_heads.0.*``, so checkpoint saving/loading and the "only mtp_heads is
trainable" guard need no changes.

Per-depth input normalization (``num_depths > 1``, shared-head mode): with one
module unrolled over K depths, depth k=0 normalizes ``h_fused`` through
``ln_h`` while depths k>=1 normalize recurrent block outputs ``z_{k-1}``
through their own ``ln_h_extra[k-1]`` LayerNorms -- each depth keeps its own
affine scale/bias instead of sharing depth 0's.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RWKV7MTPModule(nn.Module):
    """One full RWKV-7 block as the MTP head (DeepSeek-V3-style module, RWKV
    edition).

    Input fusion is shared with :class:`~mtp.heads.mlp.MTPModule`:
    ``x = proj([ln_h(h) ; ln_tok(tok_emb)])`` merges the frozen backbone's
    features with the committed-token embedding into a residual stream.  The
    stream then passes through one RWKV-7 block (``src.model.Block`` with
    layer-0 semantics), whose time-mix aggregates the fused inputs across
    positions with its own WKV state -- the draft head becomes a small
    recurrent continuation of the token stream instead of a position-local
    MLP.

    Parallel (training) form -- ``forward(h, tok_emb)`` with (B, T, C)
    inputs: depth k's input sequence is
        x_k[i] = proj([ln_h(z_{k-1,i}) ; ln_tok(emb(x_{i+k+1}))])
    (z_{-1} := backbone hidden h), and the block's state at position i
    summarizes x_k[j] for j <= i.  Causality holds because the injected
    embeddings at positions j <= i are tokens x_{j+k+1} <= x_{i+k}, all
    committed by the time depth k drafts token x_{i+k+2}.

    Recurrent (inference) form: one WKV state per depth, advanced over stream
    position j as soon as token x_{j+k+1} is committed -- implemented by
    ``spec_decode.RWKV7HeadRunner`` (the MLP head is stateless, this one is
    not, which is exactly where its extra acceptance comes from).

    ``depth`` selects the input LayerNorm under a shared head: 0 uses
    ``ln_h`` (the fused backbone anchor enters here), k >= 1 uses
    ``ln_h_extra[k-1]`` on the previous depth's recurrent output.  Single-depth
    (non-shared) heads never see ``depth > 0``.

    Requires CUDA + bf16 (the RWKV-7 fused kernels), like the backbone.
    """

    def __init__(self, args, eagle3_fusion: bool = True, num_depths: int = 1):
        super().__init__()
        from ..backbone import _rwkv_base_module  # lazy: avoids import cycle
        rw = _rwkv_base_module()  # env + kernels; also required for Block
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
        # One backbone-style block, layer_id=0: v_first = its own value, so no
        # external value-residual anchor is needed for a single-block head.
        self.block = rw.Block(args, 0)
        # Layer-0 never takes the value-residual branch (v_first = v), so
        # att.v0/v1/v2 would be dead optimizer slots; alias them onto the
        # a-pathway params exactly like the RNN inference code does for
        # backbone layer 0 (generate_data.BatchedRWKV7).
        self.block.att.v0 = self.block.att.a0
        self.block.att.v1 = self.block.att.a1
        self.block.att.v2 = self.block.att.a2
        # Depths k >= 1 get their own input LayerNorm affine (4K params each);
        # depth 0 keeps self.ln_h for the fused backbone anchor.
        self.ln_h_extra = nn.ModuleList(
            [nn.LayerNorm(C) for _ in range(max(0, int(num_depths) - 1))])
        # src.model.Block.forward returns the *unnormalized* residual stream
        # (x = x + ffn(ln2(x))); the backbone applies ln_out before the head,
        # so the head must too -- z feeds self.head(z) and the next depth.
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
        """Fuse (early, mid, post-ln_out final) backbone states into (B, T, C).

        The early/middle taps are raw residual-stream states and get their own
        LayerNorms; ``h_late`` is already normalized (it is the main head's
        input) and is used as-is.  With the ``[0, 0, I]`` init this returns
        ``h_late`` exactly at step 0.
        """
        if not self.eagle3_fusion:
            return h_late
        cat = torch.cat([
            self.ln_early(h_early),
            self.ln_mid(h_mid),
            h_late,
        ], dim=-1)
        return self.fc_fuse(cat)

    def forward(self, h: torch.Tensor, tok_emb: torch.Tensor,
                depth: int = 0) -> torch.Tensor:
        if depth and self.ln_h_extra:
            ln_h = self.ln_h_extra[depth - 1]
        else:
            ln_h = self.ln_h
        x = self.proj(torch.cat([ln_h(h), self.ln_tok(tok_emb)], dim=-1))
        z, _ = self.block(x, None)  # layer-0 block: v_first = its own value
        return self.ln_out(z)