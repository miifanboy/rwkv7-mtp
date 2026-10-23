"""
Top-k self-distillation fused with the hard-CE anchor.

The teacher is the frozen backbone's own next-token distribution, so the MTP
draft learns from the same head that will be used to verify it at inference.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .common import CE_CHUNK


def _teacher_topk(hidden: torch.Tensor, weight: torch.Tensor, k_top: int,
                  chunk: Optional[int] = None):
    """Top-k of the frozen backbone's own next-token distribution, all positions.

    ``hidden``: (B, T, C) backbone hiddens.  Returns ``(ids, probs)``, each
    (B, T, k): row ``(b, t)`` is the backbone's top-``k`` token ids and their
    renormalised probabilities at position ``t`` (i.e. its distribution for the
    token *after* ``t``).

    This is computed ONCE per micro-step and shared by every MTP depth: the
    teacher for depth k at row i is the backbone distribution at position
    ``i+k+1`` -- just a row-shift of this table.  One full-vocab GEMM pass
    instead of K.  Chunked, no-grad, nothing (rows, V)-sized retained.
    """
    chunk = CE_CHUNK if chunk is None else chunk
    B, T, C = hidden.shape
    k_eff = max(1, min(int(k_top), weight.shape[0]))
    flat = hidden.reshape(-1, C)
    ids = torch.empty(B * T, k_eff, dtype=torch.long, device=hidden.device)
    q = torch.empty(B * T, k_eff, dtype=torch.float32, device=hidden.device)
    with torch.no_grad():
        for start in range(0, B * T, chunk):
            end = min(start + chunk, B * T)
            logits = F.linear(flat[start:end], weight).float()
            vals, idxs = torch.topk(logits, k_eff, dim=-1)
            ids[start:end] = idxs
            q[start:end] = torch.softmax(vals, dim=-1)
            del logits, vals, idxs
    return ids.view(B, T, k_eff), q.view(B, T, k_eff)


class _FusedDistillCE(torch.autograd.Function):
    """Hard-CE anchor + top-k self-distillation KL, without retaining (n, V).

    The teacher distribution is supplied PRECOMPUTED: ``t_ids``/``t_q`` are
    ``(n, k)`` tensors aligned with the ``z``/``targets`` rows (slices of the
    shared per-position table from ``_teacher_topk``).  The draft hidden ``z``
    is trained toward the teacher's top-k distribution while the hard-label CE
    anchors argmax sharpness:

        L = CE(z @ W.T, targets) + lam * sum_j q~_j (log q~_j - log p_j)

    where ``p = softmax(z @ W.T)`` and the KL runs over the teacher's top-k
    support.  ``ce_w`` scales (or, with 0.0, removes) the hard-CE anchor:
    ``ce_w=0`` trains on ``lam * KL(top-k)`` alone.  Rows with
    ``targets == ignore_index`` contribute to neither term and are excluded
    from the normaliser.

    Memory discipline (same as ``_FusedHeadCE``): ``forward`` runs chunk-wise
    and saves ``z`` for backward; ``backward`` recomputes the draft logits
    chunk-wise.  Nothing ``(n, V)``-sized survives a chunk iteration -- a naive
    autograd implementation would retain ~1 GB of log-softmax per depth at
    ctx 4096.

    Gradient: with ``p`` the full-vocab softmax of the draft chunk,

        dL/dlogits = (ce_w + lam) * p - ce_w * onehot(targets) - lam * Q,
        Q = q~ scattered at the teacher top-k ids   (uses sum(q~) = 1)

    i.e. the CE gradient ``ce_w * (p - onehot)`` plus the KL gradient
    ``lam * (p - Q)``.
    The teacher table is treated as a constant (the backbone is frozen -- no
    gradient pushes the teacher toward the draft).
    """

    @staticmethod
    def forward(ctx, z, weight, targets, ignore_index, lam, t_ids, t_q, chunk,
                ce_w=1.0):
        flat_z = z.reshape(-1, z.size(-1))
        tgt = targets.reshape(-1)
        N = flat_z.shape[0]
        lam = float(lam)
        if lam != 0.0:
            if t_ids is None or t_q is None:
                raise ValueError("_FusedDistillCE: lam > 0 requires a teacher table")
            if t_ids.shape[0] != N or t_q.shape[0] != N:
                raise ValueError(
                    f"_FusedDistillCE: teacher rows {t_ids.shape[0]} != z rows {N}"
                )

        ce_sum = torch.zeros((), dtype=torch.float32, device=z.device)
        kl_sum = torch.zeros((), dtype=torch.float32, device=z.device)
        n_valid = 0
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            logits = F.linear(flat_z[start:end], weight).float()
            tgt_c = tgt[start:end]
            ce_sum = ce_sum + F.cross_entropy(
                logits, tgt_c, ignore_index=ignore_index, reduction="sum"
            )
            if lam != 0.0:
                lse = torch.logsumexp(logits, dim=-1, keepdim=True)
                logp_topk = logits.gather(1, t_ids[start:end]) - lse
                q = t_q[start:end] * (tgt_c != ignore_index).unsqueeze(1)
                kl_sum = kl_sum + (
                    q * (q.clamp_min(1e-12).log() - logp_topk)
                ).sum()
                del lse, logp_topk, q
            n_valid += int((tgt_c != ignore_index).sum())
            del logits

        ctx.lam = lam
        ctx.chunk = chunk
        ctx.ignore_index = ignore_index
        ctx.denom = max(n_valid, 1)
        ctx.z_shape = z.shape
        ctx.ids = t_ids
        ctx.q = t_q
        ctx.save_for_backward(z, weight, tgt)

        loss = (float(ce_w) * ce_sum + lam * kl_sum) / ctx.denom
        kl = kl_sum / ctx.denom
        ctx.mark_non_differentiable(kl)
        ctx.ce_w = float(ce_w)
        return loss, kl

    @staticmethod
    def backward(ctx, grad_output, _grad_kl=None):
        z, weight, tgt = ctx.saved_tensors
        flat = z.reshape(-1, z.size(-1))
        chunk, lam = ctx.chunk, ctx.lam
        ce_w = ctx.ce_w
        ignore_index = ctx.ignore_index

        grad_flat = torch.empty_like(flat)
        want_weight_grad = ctx.needs_input_grad[1]
        grad_weight = (
            torch.zeros(weight.shape, dtype=torch.float32, device=weight.device)
            if want_weight_grad else None
        )
        scale = (grad_output.reshape(()) / ctx.denom).to(torch.float32)

        for start in range(0, flat.shape[0], chunk):
            end = min(start + chunk, flat.shape[0])
            logits = F.linear(flat[start:end], weight).float()
            p = torch.softmax(logits, dim=-1)                    # (n, V) fp32
            p = p.mul_(ce_w + lam)                               # CE + lam*KL share p
            tgt_c = tgt[start:end]
            ignored = tgt_c == ignore_index
            if ce_w != 0.0:
                p.scatter_add_(1, tgt_c.masked_fill(ignored, 0).unsqueeze(1),
                               p.new_full((p.size(0), 1), -1.0))  # -ce_w*onehot
            if lam != 0.0 and ctx.ids is not None:
                p.scatter_add_(1, ctx.ids[start:end],
                               (-lam) * ctx.q[start:end].to(p.dtype))  # -lam*Q
            if ignored.any():
                p = p.masked_fill(ignored.unsqueeze(1), 0.0)
            if want_weight_grad:
                grad_weight += p.transpose(0, 1) @ flat[start:end].to(p.dtype)
            grad_flat[start:end] = (p.to(weight.dtype) @ weight).to(grad_flat.dtype)
            del logits, p

        grad_z = grad_flat.mul_(scale.to(grad_flat.dtype)).reshape(ctx.z_shape)
        if want_weight_grad:
            return (grad_z, (grad_weight * scale).to(weight.dtype), None,
                    None, None, None, None, None, None)
        return grad_z, None, None, None, None, None, None, None, None


def _distill_ce(z: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor,
                ignore_index: int, lam: float, t_ids: torch.Tensor,
                t_q: torch.Tensor, chunk: Optional[int] = None,
                ce_w: float = 1.0):
    """Fused hard-CE + top-k self-distillation KL (see ``_FusedDistillCE``).

    ``t_ids``/``t_q``: (n, k) teacher top-k table rows aligned with ``z`` --
    slice them from ``_teacher_topk(hidden, weight, k)`` per depth.
    ``ce_w=0.0`` drops the hard-CE anchor (pure ``lam * KL`` training).
    Returns ``(loss, kl)`` -- ``kl`` is detached (monitoring only).
    """
    chunk = CE_CHUNK if chunk is None else chunk
    loss, kl = _FusedDistillCE.apply(z, weight, targets,
                                     int(ignore_index), float(lam),
                                     t_ids, t_q, chunk, float(ce_w))
    return loss, kl.detach()
