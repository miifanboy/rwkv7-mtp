"""
LK losses (speculative-decoding acceptance objectives).

Reference: ``docs/lk-loss.md``.  For the draft (MTP depth) distribution q and
the frozen backbone teacher distribution p at the aligned position, the
speculative-decoding acceptance rate of a draft token is

    alpha = sum_x min(p(x), q(x)) = 1 - TV(p, q)

so the two LK losses optimize exactly what the draft is used for, where plain
KL is only a proxy:

    hybrid   :  L = lam * KL(p||q) + (1-lam) * TV(p,q)
                lam = exp(-eta * mean_row(alpha))   (stop-gradient; per call)
    lk_alpha :  L = -log(alpha + eps)

The MTP head shares the frozen backbone's full-vocab head, so there is no
draft-vocabulary truncation: p needs no renormalisation and min(p, q) runs
over the full vocabulary.

Memory discipline (same as ``_FusedHeadCE`` / ``_FusedDistillCE``): forward and
backward both recompute the draft AND the teacher projection chunk-wise; the
retained rows are only z, the aligned teacher hiddens and the targets.
Nothing (n, V)-sized survives a chunk iteration.

Closed forms used (s = sign(q - p) in {-1, 0, +1}):

    alpha_row = sum min(p, q) = 1 - TV(p, q)      (values; relu(q-p).sum == TV)
    d KL/dz_q      = q - p
    d TV/dz_q      = 0.5 * q * (s - E_q[s]),   E_q[s] = sum(q * s)
    d L_alpha/dz_q = 0.5 * q * (s - E_q[s]) / (alpha_row + eps)

(Note q * s is NOT relu(q - p): where q < p, q * s = -q.  The relu identity
only holds for the TV *value* sum, never for the gradient.)

Rows whose target is ``ignore_index`` contribute to neither loss, normaliser
nor gradient (same convention as the CE/distill paths).
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F

from ..env import _THIS_DIR, _patch_cpp_extension_flags
from .common import CE_CHUNK

LK_EPS = 1e-8

# Fused CUDA kernels for the LK elementwise chains (optional, fast path).
# The eager LK forward/backward runs ~15 full-vocab fp32 elementwise passes per
# depth; cuda/lk_loss.cu collapses them into block-per-row sweeps over the bf16
# GEMM outputs (identical inputs, fp32 math inside).  Loaded lazily on the
# first CUDA LK call; on any build failure we fall back to the eager path
# (which is also the CPU path used by the unit tests).  RWKV_LK_CUDA=0 forces
# the eager path.
_LK_EXT = None
_LK_EXT_TRIED = False


def _lk_cuda_ext():
    """Lazily build/load the LK CUDA extension (None = eager fallback)."""
    global _LK_EXT, _LK_EXT_TRIED
    if _LK_EXT_TRIED:
        return _LK_EXT
    _LK_EXT_TRIED = True
    if os.environ.get("RWKV_LK_CUDA", "1") != "1":
        return None
    if not torch.cuda.is_available():
        return None
    try:
        from torch.utils import cpp_extension
        _patch_cpp_extension_flags()   # idempotent; may already be applied
        _LK_EXT = cpp_extension.load(
            name="lk_loss_ext",
            sources=[os.path.join(_THIS_DIR, "cuda", "lk_loss.cpp"),
                     os.path.join(_THIS_DIR, "cuda", "lk_loss.cu")],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
        print("[mtp] LK CUDA kernels loaded (fused full-vocab elementwise)")
    except Exception as e:
        print(f"[mtp] LK CUDA kernel build failed ({e!r}); "
              f"falling back to the eager LK path")
        _LK_EXT = None
    return _LK_EXT


def _teacher_logits(hidden: torch.Tensor, weight: torch.Tensor,
                    chunk: Optional[int] = None) -> torch.Tensor:
    """Full-vocab teacher logits table ``(B*T, V)``, same dtype as ``hidden``.

    One full-vocab GEMM pass over all positions, chunked, under ``no_grad``:
    the LK analogue of ``_teacher_topk``.  Every MTP depth's teacher
    distribution is a row-shift of this table, so ONE pass replaces the
    per-depth teacher GEMMs in _FusedLK's forward AND backward (which is what
    makes LK ~40% slower than the top-k distill path otherwise).  Storing the
    bf16 GEMM output and upcasting per chunk later is bit-identical to
    recomputing ``.float()`` logits per chunk.
    """
    chunk = CE_CHUNK if chunk is None else chunk
    with torch.no_grad():
        h2d = hidden.reshape(-1, hidden.size(-1))
        out = torch.empty(h2d.shape[0], weight.shape[0],
                          dtype=h2d.dtype, device=hidden.device)
        for start in range(0, h2d.shape[0], chunk):
            end = min(start + chunk, h2d.shape[0])
            out[start:end] = F.linear(h2d[start:end], weight)
    return out


class _FusedLK(torch.autograd.Function):
    """LK loss of one MTP depth vs the frozen teacher, without retaining (n, V).

    Inputs (all row-aligned over the flattened (B, T_rows) grid):

        z        (B, T_rows, C)  draft hidden for this depth (valid rows only)
        weight   (V, C)          the shared frozen head weight
        targets  (B, T_rows)     next-token labels; ``ignore_index`` rows are
                                 excluded from loss, normaliser and gradients
        t        teacher source, per ``t_mode``:
                   "hidden": (B, T_rows, C) frozen backbone hiddens at the
                             positions whose next token each draft row predicts
                   "logits": (B*T, V) precomputed teacher logits table
                             (_teacher_logits); flat row r = b*T_rows + i reads
                             table row (b*T + i + t_offset)

    Losses (fp32 softmaxes over the full vocabulary, per row):

        alpha_row = sum min(p, q) = 1 - TV(p, q)
        mode "hybrid":   L = lam * KL(p||q) + (1-lam) * TV(p,q)
                         lam = exp(-eta * mean_valid(alpha)).detach()
        mode "lk_alpha": L = -log(alpha_row + eps)

    Returns ``(loss, alpha_mean, kl_mean, tv_mean)``; the monitors are detached
    statistics for logging (kl_mean / tv_mean are 0-valued constants in
    "lk_alpha" mode, where only alpha is meaningful).
    """

    @staticmethod
    def forward(ctx, z: torch.Tensor, weight: torch.Tensor,
                targets: torch.Tensor, t: torch.Tensor, t_mode: str,
                t_offset: int, t_T: int, ignore_index: int, mode: str,
                eta: float, eps: float, chunk: int):
        flat_z = z.reshape(-1, z.size(-1))
        tgt = targets.reshape(-1)
        N = flat_z.shape[0]
        B, n_rows = z.shape[0], z.shape[1]
        if tgt.numel() != N:
            raise ValueError(
                f"_FusedLK: {N} draft rows but {tgt.numel()} targets; "
                f"inputs must be row-aligned")
        if t_mode not in ("hidden", "logits"):
            raise ValueError(f"_FusedLK: unknown t_mode {t_mode!r}")
        if t_mode == "logits":
            if t.shape[0] != B * t_T:
                raise ValueError(
                    f"_FusedLK: logits table has {t.shape[0]} rows, expected "
                    f"B*t_T = {B * t_T}")
        elif t.reshape(N, -1).shape[0] != N:
            raise ValueError(
                f"_FusedLK: teacher hiddens rows {t.reshape(N, -1).shape[0]} "
                f"!= draft rows {N}")
        if mode not in ("hybrid", "lk_alpha"):
            raise ValueError(f"_FusedLK: unknown mode {mode!r}")

        def teacher_p_logits(start: int, end: int) -> torch.Tensor:
            # bf16, like the raw GEMM/table output -- the eager path upcasts
            # itself; the CUDA kernels read bf16 directly.
            if t_mode == "hidden":
                return F.linear(ft_h[start:end], weight)
            # logits table: gather this chunk's rows (batch-interleaved)
            r = torch.arange(start, end, device=z.device)
            gidx = (r // n_rows) * t_T + (r % n_rows) + t_offset
            return t[gidx]

        ext = (_lk_cuda_ext() if (z.is_cuda and flat_z.dtype == torch.bfloat16)
               else None)
        hybrid = mode == "hybrid"

        ft_h = t.reshape(N, -1) if t_mode == "hidden" else None

        alpha_sum = torch.zeros((), dtype=torch.float32, device=z.device)
        kl_sum = torch.zeros((), dtype=torch.float32, device=z.device)
        tv_sum = torch.zeros((), dtype=torch.float32, device=z.device)
        nl_sum = torch.zeros((), dtype=torch.float32, device=z.device)
        n_valid = 0
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            valid = tgt[start:end] != ignore_index
            nv = int(valid.sum())
            n_valid += nv
            if nv == 0:
                continue
            if ext is not None:
                # Fused kernel: one block-per-row sweep over the bf16 logits.
                q_logits = F.linear(flat_z[start:end], weight)
                p_logits = teacher_p_logits(start, end)
                alpha_row, kl_row = ext.lk_forward_stats(
                    q_logits, p_logits, hybrid, float(eps))
                del q_logits, p_logits
            else:
                q_logits = F.linear(flat_z[start:end], weight).float()
                p_logits = teacher_p_logits(start, end).float()
                logq = F.log_softmax(q_logits, dim=-1)
                del q_logits
                logp = F.log_softmax(p_logits, dim=-1)
                del p_logits
                q = logq.exp()
                p = logp.exp()
                A = q.sub(p).clamp_min_(0)          # relu(q - p), sums to TV
                tv_row = A.sum(dim=-1)              # (n,)  TV = 1 - alpha
                alpha_row = 1.0 - tv_row
                if hybrid:
                    # KL(p || q) = sum p (log p - log q); F.kl_div handles
                    # p == 0 (numerically underflowed) as a zero contribution.
                    kl_row = F.kl_div(logq, p, reduction="none").sum(dim=-1)
                else:
                    kl_row = None
                del logq, logp, q, p, A, tv_row
            alpha_sum = alpha_sum + alpha_row[valid].sum()
            tv_sum = tv_sum + (1.0 - alpha_row)[valid].sum()
            if hybrid:
                kl_sum = kl_sum + kl_row[valid].sum()
                del kl_row
            else:
                nl_sum = nl_sum + (-(alpha_row[valid] + eps).log()).sum()
            del alpha_row

        denom = max(n_valid, 1)
        alpha_mean = alpha_sum / denom
        kl_mean = kl_sum / denom
        tv_mean = tv_sum / denom
        if mode == "hybrid":
            # Trust-region schedule: stop-gradient on alpha (forward runs
            # under no_grad anyway, so alpha_mean is already a constant).
            lam = float(torch.exp(-float(eta) * alpha_mean))
            loss = (lam * kl_sum + (1.0 - lam) * tv_sum) / denom
        else:
            lam = 0.0
            loss = nl_sum / denom

        ctx.mark_non_differentiable(alpha_mean, kl_mean, tv_mean)
        ctx.mode = mode
        ctx.lam = lam
        ctx.eps = float(eps)
        ctx.chunk = chunk
        ctx.ignore_index = ignore_index
        ctx.denom = denom
        ctx.z_shape = z.shape
        ctx.t_mode, ctx.t_offset, ctx.t_T = t_mode, int(t_offset), int(t_T)
        ctx.B, ctx.n_rows = B, n_rows
        # backward recomputes the draft projection chunk-wise and slices the
        # teacher source (shared table or hiddens) -- nothing (n, V)-sized is
        # retained beyond what the caller already holds.
        ctx.save_for_backward(z, weight, tgt, t)
        return loss, alpha_mean, kl_mean, tv_mean

    @staticmethod
    def backward(ctx, grad_output, _g_alpha, _g_kl, _g_tv):
        z, weight, tgt, t = ctx.saved_tensors
        flat = z.reshape(-1, z.size(-1))
        chunk, mode, lam = ctx.chunk, ctx.mode, ctx.lam
        eps, ignore_index = ctx.eps, ctx.ignore_index
        N = flat.shape[0]
        t_mode, t_offset, t_T = ctx.t_mode, ctx.t_offset, ctx.t_T
        B, n_rows = ctx.B, ctx.n_rows
        ft_h = t.reshape(N, -1) if t_mode == "hidden" else None

        def teacher_p_logits(start: int, end: int) -> torch.Tensor:
            # Must return bf16 (same convention as the forward closure): the
            # eager path upcasts itself; the CUDA kernels read bf16 directly.
            if t_mode == "hidden":
                return F.linear(ft_h[start:end], weight)
            r = torch.arange(start, end, device=z.device)
            gidx = (r // n_rows) * t_T + (r % n_rows) + t_offset
            return t[gidx]

        ext = (_lk_cuda_ext() if (z.is_cuda and flat.dtype == torch.bfloat16)
               else None)
        hybrid = mode == "hybrid"

        grad_flat = torch.empty_like(flat)
        want_weight_grad = ctx.needs_input_grad[1]
        grad_weight = (
            torch.zeros(weight.shape, dtype=torch.float32, device=weight.device)
            if want_weight_grad else None
        )
        scale = (grad_output.reshape(()) / ctx.denom).to(torch.float32)

        for start in range(0, flat.shape[0], chunk):
            end = min(start + chunk, flat.shape[0])
            valid = tgt[start:end] != ignore_index
            if not bool(valid.any()):
                grad_flat[start:end] = 0
                continue
            if ext is not None:
                # Fused kernels: stage 1 recomputes the row scalars (E_q[s],
                # alpha, log-softmax normalisers), stage 2 writes the (n, V)
                # logit gradient in bf16, zeros for ignore_index rows.
                q_logits = F.linear(flat[start:end], weight)
                p_logits = teacher_p_logits(start, end)
                es, alpha_r, _, _ = ext.lk_row_scalars(q_logits, p_logits)
                g = ext.lk_grad(q_logits, p_logits, es, alpha_r,
                                tgt[start:end], ignore_index, hybrid,
                                lam, eps)
                del q_logits, p_logits, es, alpha_r
            else:
                q_logits = F.linear(flat[start:end], weight).float()
                p_logits = teacher_p_logits(start, end).float()
                logq = F.log_softmax(q_logits, dim=-1)
                del q_logits
                logp = F.log_softmax(p_logits, dim=-1)
                del p_logits
                q = logq.exp()
                del logq
                p = logp.exp()
                del logp
                diff = q.sub(p)                     # q - p  (KL gradient)
                del p
                s = torch.sign(diff)                # s = sign(q - p)
                g = q.mul(s)                        # q * s
                es = g.sum(dim=-1, keepdim=True)    # E_q[s]
                # TV gradient: 0.5 * q * (s - E_q[s]); addcmul_ avoids
                # materialising the (n, V) broadcast product q * E_q[s].
                g.addcmul_(q, es, value=-1.0).mul_(0.5)
                if mode == "lk_alpha":
                    # alpha = 1 - TV; TV value = 0.5 * sum|q - p| (relu sum).
                    tv_row = diff.abs_().sum(dim=-1, keepdim=True).mul_(0.5)
                    g.div_(1.0 - tv_row + eps)      # /(alpha + eps) per row
                else:
                    # lam * (q - p) + (1 - lam) * 0.5 * q * (s - E_q[s])
                    g.mul_(1.0 - lam).add_(diff.mul_(lam))
                del q, diff, s
                inv = ~valid
                if bool(inv.any()):
                    g.masked_fill_(inv.unsqueeze(1), 0.0)
            if want_weight_grad:
                grad_weight += g.to(torch.float32).transpose(0, 1) @ \
                    flat[start:end].to(torch.float32)
            grad_flat[start:end] = (g.to(weight.dtype) @ weight).to(
                grad_flat.dtype)
            del g

        grad_z = grad_flat.mul_(scale.to(grad_flat.dtype)).reshape(ctx.z_shape)
        if want_weight_grad:
            return (grad_z, (grad_weight * scale).to(weight.dtype), None, None,
                    None, None, None, None, None, None, None, None)
        return grad_z, None, None, None, None, None, None, None, None, None, None, None


def _lk_loss(z: torch.Tensor, t_h: Optional[torch.Tensor],
             weight: torch.Tensor, targets: torch.Tensor, ignore_index: int,
             mode: str, eta: float, t_logits: Optional[torch.Tensor] = None,
             t_offset: int = 0, t_T: int = 0, eps: float = LK_EPS,
             chunk: Optional[int] = None):
    """LK loss of one MTP depth (see ``_FusedLK``).

    Teacher source, exactly one of:
    - ``t_h``: (B, T_rows, C) frozen backbone hiddens aligned with ``z`` -- for
      depth k these are the rows ``hidden[:, k+1 : k+1+T_rows]`` (the teacher
      position whose next token draft row i predicts is i + k + 1);
    - ``t_logits``: (B*T, V) precomputed teacher logits table from
      ``_teacher_logits`` (shared across depths; ``t_offset`` = depth + 1,
      ``t_T`` = per-sequence length T).

    ``mode``: "hybrid" (adaptive KL+TV, ``eta`` schedule) or "lk_alpha"
    (-log alpha).  Returns ``(loss, alpha_mean, kl_mean, tv_mean)`` with the
    monitors detached (logging only).
    """
    chunk = CE_CHUNK if chunk is None else chunk
    if (t_logits is None) == (t_h is None):
        raise ValueError("_lk_loss: exactly one of t_h / t_logits must be given")
    if t_logits is not None:
        loss, a, kl_m, tv_m = _FusedLK.apply(
            z, weight, targets, t_logits, "logits", int(t_offset), int(t_T),
            int(ignore_index), mode, float(eta), float(eps), chunk)
    else:
        loss, a, kl_m, tv_m = _FusedLK.apply(
            z, weight, targets, t_h.detach(), "hidden", 0, 0,
            int(ignore_index), mode, float(eta), float(eps), chunk)
    return loss, a.detach(), kl_m.detach(), tv_m.detach()
