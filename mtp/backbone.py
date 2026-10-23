"""
Backbone assembly: the frozen RWKV-7 plus the chained MTP heads.

The heavy ``src.model`` import (which compiles the CUDA kernels) happens lazily
on first use, so importing :mod:`mtp` stays cheap and the pure-loss tests can
run on CPU without a backbone.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from .env import (
    _install_lightning_shim,
    _patch_cpp_extension_flags,
    _prepare_build_env,
)
from .heads import MTP_ARCHS, MTPModule, RWKV7MTPModule, build_head
from .losses import (
    _distill_ce,
    _head_ce,
    _lk_loss,
    _teacher_logits,
    _teacher_topk,
)
from .tensors import _shift_left

_RWKV_BASE = None
_RWKV_BASE_MOD = None
_RWKV_WITH_MTP = None


def _load_rwkv_base():
    global _RWKV_BASE, _RWKV_BASE_MOD
    if _RWKV_BASE is None:
        _install_lightning_shim()
        _prepare_build_env()
        _patch_cpp_extension_flags()
        from src import model as _rwkv_model  # heavy: compiles CUDA kernels
        _RWKV_BASE = _rwkv_model.RWKV
        _RWKV_BASE_MOD = _rwkv_model
    return _RWKV_BASE


def _rwkv_base_module():
    """The imported ``src.model`` module (kernels built); for head re-use."""
    _load_rwkv_base()
    assert _RWKV_BASE_MOD is not None
    return _RWKV_BASE_MOD


def _make_rwkv_with_mtp():
    global _RWKV_WITH_MTP
    if _RWKV_WITH_MTP is not None:
        return _RWKV_WITH_MTP

    RWKV = _load_rwkv_base()

    class RWKVWithMTP(RWKV):
        """RWKV-7 backbone + chained MTP head(s) on top of the last hidden state.

        ``shared_head=False`` (default, DeepSeek-V3 style): ``num_mtp``
        independent MTPModules, one per depth.

        ``shared_head=True`` (FastMTP style): a *single* MTPModule stored in
        ``mtp_heads[0]`` and applied ``num_mtp`` times recursively, so the
        training recurrence matches the inference draft loop (the head learns
        to condition on its own output hidden states).  ``num_mtp`` is then the
        draft/recursion depth, not a module count.
        """

        def __init__(self, args, num_mtp: int, shared_head: bool = False,
                     arch: str = "mlp", eagle3_fusion: bool = False):
            super().__init__(args)
            if arch not in MTP_ARCHS:
                raise ValueError(f"unknown --mtp-arch {arch!r}; choose from {MTP_ARCHS}")
            if eagle3_fusion and arch != "rwkv7":
                raise ValueError(
                    "eagle3_fusion requires --mtp-arch rwkv7 (the EAGLE-3 "
                    "multi-layer fusion parameters live in RWKV7MTPModule)")
            self.num_mtp = int(num_mtp)
            self.shared_head = bool(shared_head)
            self.mtp_arch = arch
            self.eagle3_fusion = bool(eagle3_fusion)
            self.freeze_backbone = True
            n_modules = 1 if self.shared_head else self.num_mtp
            # Shared head: one module unrolled over num_mtp depths, so the head
            # needs one input LayerNorm per depth (ln_h + ln_h_extra[k-1]).
            num_depths = self.num_mtp if self.shared_head else 1
            self.mtp_heads = nn.ModuleList([
                build_head(arch, args, eagle3_fusion=eagle3_fusion,
                           num_depths=num_depths)
                for _ in range(n_modules)
            ])

        def _head(self, k: int) -> nn.Module:
            """Module used at depth ``k`` (the same one everywhere if shared)."""
            return self.mtp_heads[0] if self.shared_head else self.mtp_heads[k]

        def _head_forward(self, mtp: nn.Module, h: torch.Tensor,
                          tok_emb: torch.Tensor, depth: int) -> torch.Tensor:
            """Call a head at depth ``depth`` (rwkv7 heads take a depth arg for
            their per-depth input LayerNorms; other heads ignore it)."""
            if depth and isinstance(mtp, RWKV7MTPModule):
                return mtp(h, tok_emb, depth=depth)
            return mtp(h, tok_emb)

        # -- hidden states -------------------------------------------------
        def hidden_states(self, idx: torch.Tensor) -> torch.Tensor:
            if self.freeze_backbone:
                with torch.no_grad():
                    return self._forward_features(idx)
            return self._forward_features(idx)

        @staticmethod
        def tap_layers(n_layer: int) -> tuple:
            """EAGLE-3 tap indices (0-indexed blocks) for an ``n_layer`` backbone:
            (early, middle, final).  The final tap is taken after ln_out."""
            return (max(1, n_layer // 8 - 1), n_layer // 2 - 1, n_layer - 1)

        def _forward_features_multi(self, idx: torch.Tensor):
            """Run the frozen backbone and return
            ``(h_early, h_mid, hidden_post_ln_out)`` -- the EAGLE-3 taps.
            Identical math to ``_forward_features`` plus two mid-stream taps;
            ``src/model.py`` (upstream) stays untouched."""
            args = self.args
            B, T = idx.size()
            assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."
            i_early, i_mid, _ = self.tap_layers(args.n_layer)

            x = self.emb(idx)
            v_first = torch.empty_like(x)
            h_early = h_mid = None
            for i, block in enumerate(self.blocks):
                if args.grad_cp == 1:
                    # deepspeed is a src.model module-level import; grad_cp is 0
                    # in this retrofit, this is only a fidelity fallback.
                    from src import model as _rwkv_model
                    x, v_first = _rwkv_model.deepspeed.checkpointing.checkpoint(
                        block, x, v_first)
                else:
                    x, v_first = block(x, v_first)
                if i == i_early:
                    h_early = x
                elif i == i_mid:
                    h_mid = x
            hidden = self.ln_out(x)
            return h_early, h_mid, hidden

        def hidden_states_with_h0(self, idx: torch.Tensor):
            """Return ``(hidden, h0_for_depth0)``.

            - ``hidden``: post-ln_out final layer -- the EXACT tensor feeding
              ``main_logits``, the teacher targets and the ``drafts[0]`` pick.
            - ``h0``: what depth k=0 receives as its state input: the EAGLE-3
              fused anchor when head 0 has fusion, else ``hidden`` (bit-identical
              to the pre-fusion behavior).
            """
            head0 = self._head(0)
            use_fusion = bool(getattr(head0, "eagle3_fusion", False))
            if not use_fusion:
                h = self.hidden_states(idx)
                return h, h
            if self.freeze_backbone:
                with torch.no_grad():
                    h_early, h_mid, hidden = self._forward_features_multi(idx)
            else:
                h_early, h_mid, hidden = self._forward_features_multi(idx)
            # fuse_backbone runs OUTSIDE no_grad so fc_fuse, ln_early and ln_mid
            # receive gradients (the taps themselves are frozen constants).
            h0 = head0.fuse_backbone(h_early, h_mid, hidden)
            return hidden, h0

        # -- eval / generic forward (returns per-depth logits) -------------
        def forward(self, idx: torch.Tensor, return_mtp: bool = False):
            hidden, h0 = self.hidden_states_with_h0(idx)
            main_logits = self.head(hidden)
            if not return_mtp:
                return main_logits

            mtp_logits: List[torch.Tensor] = []
            h_prev = h0
            for k in range(self.num_mtp):
                mtp = self._head(k)
                tok = self.emb(_shift_left(idx, k + 1, pad=0))
                z = self._head_forward(mtp, h_prev, tok, k if self.shared_head else 0)
                mtp_logits.append(self.head(z))
                h_prev = z
            return main_logits, mtp_logits

        # -- memory-bounded training loss ----------------------------------
        def forward_loss(self, idx, targets, decay: float = 0.8,
                         ignore_index: int = 0, return_parts: bool = False,
                         include_main: bool = True,
                         distill_topk: int = 0, distill_lambda: float = 0.5,
                         kl_only: bool = False,
                         lk_mode: str = "", lk_eta: float = 3.0,
                         lk_teacher_cache: str = "auto"):
            """Compute the weighted loss one depth at a time.

            Every depth goes through ``_head_ce``, which fuses the shared head
            projection with the cross-entropy and recomputes the projection in
            its backward pass.  No ``(B, T, V)`` tensor is ever materialised, so
            nothing T-proportional survives into the backward pass -- that is
            what lets ctx 4096 x K=3 fit in 12 GB (the old logits-based path
            retained ~1 GiB of fp32 log-softmax *per depth*).

            ``include_main=False`` drops the main-head CE.  That term is the
            frozen backbone head applied to the frozen backbone hidden state, so
            it has no ``grad_fn`` at all: it costs ~3.5% of the step and
            contributes exactly zero gradient.  It is kept by default purely as
            an in-distribution monitor.  ``total_w`` counts its weight either
            way, so skipping it leaves the MTP gradient bit-identical.

            ``distill_topk > 0`` (and ``distill_lambda > 0``) adds top-k
            self-distillation to the MTP depth losses: for depth k, the teacher
            is the frozen backbone's own distribution at position ``i+k+1`` (the
            position whose next token is depth k's target), restricted to its
            top-``distill_topk`` tokens and renormalised.  Depth loss becomes
            ``CE(hard) + distill_lambda * KL(top-k)``, computed by
            ``_FusedDistillCE`` under the same memory discipline.  The teacher
            top-k table is computed ONCE per micro-step over all positions
            (``_teacher_topk``) and row-shift-sliced per depth -- one
            full-vocab teacher GEMM pass instead of K.  The main head is
            unaffected (it is a zero-grad monitor).  Only the rows with a real
            target (``i <= T-k-3``) take part; the KL for the rest would be
            undefined anyway.

            ``lk_mode="hybrid"|"lk_alpha"`` replaces the whole depth loss with
            an LK loss (docs/lk-loss.md) against the frozen backbone teacher at
            the aligned position, over the FULL vocabulary (the shared head means
            no draft-vocab truncation, so no target renormalisation is needed):
              alpha = sum min(p, q) = 1 - TV(p, q)  -- the acceptance rate
              "hybrid"  : lam * KL(p||q) + (1-lam) * TV(p,q),
                          lam = exp(-lk_eta * mean alpha) per depth per call
              "lk_alpha": -log(alpha)
            ``distill_topk`` / ``kl_only`` must be off in this mode; the
            teacher probabilities come from full-vocab recomputation rather
            than the top-k table (TV/alpha need the draft's full support).
            ``parts["alpha_k"]`` reports the mean acceptance estimate per depth
            -- the direct deployment metric this loss optimizes.
            """
            hidden, h0 = self.hidden_states_with_h0(idx)

            loss = None
            parts: Dict[str, torch.Tensor] = {}
            if include_main:
                ce_main = _head_ce(hidden, self.head.weight, targets, ignore_index)
                loss = ce_main
                parts["main"] = ce_main.detach()
            total_w = 1.0

            T = targets.shape[1]
            if lk_mode not in ("", "hybrid", "lk_alpha"):
                raise ValueError(f"forward_loss: unknown lk_mode {lk_mode!r}")
            use_lk = lk_mode != ""
            if use_lk and (distill_topk > 0 or kl_only):
                raise ValueError(
                    "forward_loss: lk_mode replaces the top-k distill path; "
                    "pass distill_topk=0 and kl_only=False")
            use_distill = distill_topk > 0 and distill_lambda > 0 and T > 2
            if kl_only and not use_distill:
                raise ValueError(
                    "forward_loss: kl_only=True requires distill_topk > 0, "
                    "distill_lambda > 0 and seq len > 2")
            t_ids = t_q = None
            if use_distill:
                t_ids, t_q = _teacher_topk(hidden, self.head.weight, distill_topk)

            # LK teacher source.  "auto": cache the full-vocab teacher logits
            # table once per micro-step (one GEMM instead of per-depth GEMMs in
            # forward AND backward) when the bf16/fp32 buffer fits in ~1 GiB.
            # "off" falls back to per-depth teacher-hiddens recompute.
            t_logits = None
            if use_lk:
                if lk_teacher_cache not in ("auto", "on", "off"):
                    raise ValueError(
                        f"forward_loss: unknown lk_teacher_cache "
                        f"{lk_teacher_cache!r}")
                N_all, V = hidden.shape[0] * hidden.shape[1], self.head.weight.shape[0]
                cache_bytes = N_all * V * hidden.element_size()
                use_cache = (lk_teacher_cache == "on" or
                             (lk_teacher_cache == "auto" and cache_bytes <= 1 << 30))
                if use_cache:
                    t_logits = _teacher_logits(hidden, self.head.weight)

            h_prev = h0
            for k in range(self.num_mtp):
                mtp = self._head(k)
                tok = self.emb(_shift_left(idx, k + 1, pad=0))
                z = self._head_forward(mtp, h_prev, tok, k if self.shared_head else 0)
                tgt_k = _shift_left(targets, k + 1, pad=ignore_index)
                n_rows = T - (k + 2)          # rows with a real target token
                if use_lk and n_rows > 0:
                    z_v = z[:, :n_rows, :]
                    tgt_v = tgt_k[:, :n_rows]
                    # teacher for depth-k row i = backbone position (i+k+1)
                    if t_logits is not None:
                        ce_k, alpha_k, kl_k, tv_k = _lk_loss(
                            z_v, None, self.head.weight, tgt_v, ignore_index,
                            lk_mode, lk_eta, t_logits=t_logits,
                            t_offset=k + 1, t_T=T,
                        )
                    else:
                        t_h = hidden[:, k + 1:k + 1 + n_rows, :]
                        ce_k, alpha_k, kl_k, tv_k = _lk_loss(
                            z_v, t_h, self.head.weight, tgt_v, ignore_index,
                            lk_mode, lk_eta,
                        )
                    parts[f"alpha_{k + 1}"] = alpha_k
                    if lk_mode == "hybrid":
                        parts[f"kl_{k + 1}"] = kl_k
                elif use_lk:
                    # Degenerate rows only (T too short for this depth): zero
                    # loss, keep the graph connected.
                    ce_k = z.sum() * 0.0
                elif use_distill and n_rows > 0:
                    z_v = z[:, :n_rows, :]
                    tgt_v = tgt_k[:, :n_rows]
                    # teacher for depth-k row i = table row (i+k+1), per sequence
                    ids_k = t_ids[:, k + 1:k + 1 + n_rows].reshape(-1, t_ids.shape[-1])
                    q_k = t_q[:, k + 1:k + 1 + n_rows].reshape(-1, t_q.shape[-1])
                    ce_k, kl_k = _distill_ce(
                        z_v, self.head.weight, tgt_v, ignore_index,
                        distill_lambda, ids_k, q_k,
                        ce_w=0.0 if kl_only else 1.0,
                    )
                    parts[f"kl_{k + 1}"] = kl_k
                elif kl_only:
                    # Degenerate rows only (T too short for this depth): zero
                    # loss, keep the graph connected.
                    ce_k = z.sum() * 0.0
                else:
                    ce_k = _head_ce(z, self.head.weight, tgt_k, ignore_index)
                w = decay ** (k + 1)
                loss = w * ce_k if loss is None else loss + w * ce_k
                total_w += w
                parts[f"mtp_{k + 1}"] = ce_k.detach()
                h_prev = z

            if loss is None:
                raise ValueError("forward_loss needs include_main=True or >=1 MTP head")
            loss = loss / total_w
            parts["total"] = loss.detach()
            if return_parts:
                return loss, parts
            return loss

    RWKVWithMTP.__name__ = "RWKVWithMTP"
    RWKVWithMTP.__qualname__ = "RWKVWithMTP"
    _RWKV_WITH_MTP = RWKVWithMTP
    return RWKVWithMTP


__all__ = [
    "RWKVWithMTP",
    "_load_rwkv_base",
    "_rwkv_base_module",
    "_make_rwkv_with_mtp",
    "MTPModule",
    "RWKV7MTPModule",
]
