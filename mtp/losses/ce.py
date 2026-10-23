"""
Fused head-projection + cross-entropy (no ``(B, T, V)`` tensor materialised).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .common import CE_CHUNK


class _FusedHeadCE(torch.autograd.Function):
    """``cross_entropy(z @ weight.T, targets)`` without ever building logits.

    ``z`` is (B, T, C) and ``weight`` is (V, C) -- normally the frozen backbone
    ``head.weight``.  Saving the (B, T, V) log-softmax is what makes the naive
    path cost ~1 GiB per depth, so instead only ``z`` is saved and the
    projection is recomputed chunk-wise in ``backward``.  The projection GEMM
    therefore runs twice; it is memory-bound, so that is cheap next to the
    ~1 GiB per depth it frees.

    ``weight`` is assumed frozen (the whole retrofit freezes the backbone).  If
    it does need grad the weight gradient is accumulated chunk-wise, but note
    that needs a (V, C) fp32 buffer -- most of what this class exists to avoid.
    """

    @staticmethod
    def forward(ctx, z: torch.Tensor, weight: torch.Tensor,
                targets: torch.Tensor, ignore_index: int,
                chunk: int) -> torch.Tensor:
        flat = z.reshape(-1, z.size(-1))
        tgt = targets.reshape(-1)
        if tgt.numel() != flat.size(0):
            raise ValueError(
                f"_FusedHeadCE: {flat.size(0)} hidden rows but {tgt.numel()} "
                f"targets; targets must be broadcastable to z.shape[:-1]"
            )

        loss_sum = torch.zeros((), dtype=torch.float32, device=z.device)
        n_valid = 0
        for start in range(0, flat.size(0), chunk):
            end = min(start + chunk, flat.size(0))
            # bf16 projection then fp32 upcast: exactly what head(z).float() did.
            logits = F.linear(flat[start:end], weight).float()
            loss_sum = loss_sum + F.cross_entropy(
                logits, tgt[start:end], ignore_index=ignore_index, reduction="sum",
            )
            n_valid += int((tgt[start:end] != ignore_index).sum())

        ctx.ignore_index = ignore_index
        ctx.chunk = chunk
        ctx.denom = max(n_valid, 1)
        ctx.z_shape = z.shape
        # ``forward`` runs with grad disabled, so no graph is built here and
        # each chunk's logits die at the end of its iteration.  Only what
        # backward needs to recompute them is retained.
        if z.requires_grad or weight.requires_grad:
            ctx.save_for_backward(z, weight, tgt)
        return loss_sum / ctx.denom

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        z, weight, tgt = ctx.saved_tensors
        flat = z.reshape(-1, z.size(-1))
        chunk = ctx.chunk
        ignore_index = ctx.ignore_index

        grad_z = torch.empty_like(flat)
        want_weight_grad = ctx.needs_input_grad[1]
        grad_weight = (
            torch.zeros(weight.shape, dtype=torch.float32, device=weight.device)
            if want_weight_grad else None
        )
        scale = (grad_output.reshape(()) / ctx.denom).to(torch.float32)

        for start in range(0, flat.size(0), chunk):
            end = min(start + chunk, flat.size(0))
            logits = F.linear(flat[start:end], weight).float()
            p = torch.softmax(logits, dim=-1)                    # (n, V) fp32
            tgt_c = tgt[start:end]
            ignored = tgt_c == ignore_index
            # d/dlogits of -log p[y] is (p - onehot(y)); ignored rows are then
            # zeroed so they contribute nothing to either gradient.
            p.scatter_add_(1, tgt_c.masked_fill(ignored, 0).unsqueeze(1),
                           p.new_full((p.size(0), 1), -1.0))
            if ignored.any():
                p = p.masked_fill(ignored.unsqueeze(1), 0.0)
            if want_weight_grad:
                grad_weight += p.transpose(0, 1) @ flat[start:end].to(p.dtype)
            grad_z[start:end] = (p.to(weight.dtype) @ weight).to(grad_z.dtype)

        grad_z = grad_z.mul_(scale.to(grad_z.dtype)).reshape(ctx.z_shape)
        if want_weight_grad:
            return grad_z, (grad_weight * scale).to(weight.dtype), None, None, None
        return grad_z, None, None, None, None


def _head_ce(z: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor,
             ignore_index: int, chunk: Optional[int] = None) -> torch.Tensor:
    """Memory-bounded ``cross_entropy(z @ weight.T, targets)``.

    ``z``: (B, T, C); ``weight``: (V, C); ``targets``: broadcastable to
    ``z.shape[:-1]``.  See ``_FusedHeadCE``.
    """
    chunk = CE_CHUNK if chunk is None else chunk
    return _FusedHeadCE.apply(z, weight, targets, ignore_index, chunk)
