"""
Position-local MLP MTP head (the original, DeepSeek-V3-inspired module).

Architecture of one MTP head::

    z_k = MTPModule_k(h_{k-1}, Emb(x[i+k+1]))
    logits_k = head(z_k)                       # head is the frozen backbone head

    MTPModule(h, tok) = proj([LN(h), LN(tok)]) + mlp(LN(proj(...)))

The heads are *chained*: depth k feeds its output hidden into depth k+1
(h_0 = backbone hidden).  This is the standard DeepSeek MTP recurrence and lets
deep heads condition on the intermediate predictions.

Shift convention (IMPORTANT -- unit-tested in ``tests/test_mtp_logic.py``)
-------------------------------------------------------------------------
Let ``x[0..T-1]`` be the token ids and ``h[i]`` the backbone hidden at i.
The dataset provides next-token labels ``targets`` with ``targets[i] = x[i+1]``.

    main head       at i predicts x[i+1] = targets[i]

    MTP depth k     (k = 0 .. K-1):
        input token  at i = x[i+k+1]     = _shift_left(idx,     k+1)[i]
        target token at i = x[i+k+2]     = _shift_left(targets, k+1)[i]

So depth 0 predicts 2 tokens ahead, depth k predicts k+2 tokens ahead.
The input token is exactly one step before the target, which is what the
chained recurrence needs.  ``_shift_left(t, s)[i] = t[i+s]``.

NOTE: the handoff text mentions ``_shift_left(targets, k - 1)`` for the input
token.  With the target rule it also states (``targets[i+k+1]``) that would be
off by one and would skip the immediately-preceding token.  This package uses
the self-consistent convention above; ``tests/test_mtp_logic.py`` derives and
asserts it.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MTPModule(nn.Module):
    """proj + residual MLP + LayerNorm.  No attention / no full block."""

    def __init__(self, n_embd: int, hidden_mult: float = 2.0, eps: float = 1e-5,
                 use_mlp: bool = True):
        super().__init__()
        hidden = max(32, int(hidden_mult * n_embd) // 32 * 32)
        self.n_embd = n_embd
        self.use_mlp = use_mlp
        self.ln_h = nn.LayerNorm(n_embd, eps=eps)
        self.ln_tok = nn.LayerNorm(n_embd, eps=eps)
        self.proj = nn.Linear(2 * n_embd, n_embd, bias=False)
        if use_mlp:
            self.mlp = nn.Sequential(
                nn.LayerNorm(n_embd, eps=eps),
                nn.Linear(n_embd, hidden, bias=False),
                nn.SiLU(),
                nn.Linear(hidden, n_embd, bias=False),
            )
        else:
            self.mlp = None
        self._init_weights()

    def _init_weights(self) -> None:
        # Start close to identity-ish: the MLP residual branch is zero at init,
        # so the head begins as a learned linear mix of (hidden, token).
        nn.init.normal_(self.proj.weight, std=0.02)
        if self.mlp is not None:
            nn.init.normal_(self.mlp[1].weight, std=0.02)
            nn.init.zeros_(self.mlp[-1].weight)

    def forward(self, h: torch.Tensor, tok_emb: torch.Tensor) -> torch.Tensor:
        x = self.proj(torch.cat([self.ln_h(h), self.ln_tok(tok_emb)], dim=-1))
        if self.mlp is not None:
            x = x + self.mlp(x)
        return x
