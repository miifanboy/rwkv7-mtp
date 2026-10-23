# LK Losses — Mathematics and Implementation Logic

> Focus: the mathematical definitions and logic needed to implement **LK losses** for speculative decoding. Benchmark tables, evaluation protocols, and experimental results are omitted.

## 1. Speculative Decoding Setup

Let:

- \(p\) be the target model distribution.
- \(q\) be the draft model distribution.
- \(z_q\) be the draft logits, with \(q = \mathrm{softmax}(z_q)\).
- \(z_p\) be the target logits, with \(p = \mathrm{softmax}(z_p)\).
- \(\mathcal{V}\) be the token vocabulary.

A draft token \(x\) proposed from context \(\mathbf{c}\) is accepted with probability

\[
\beta(x \mid \mathbf{c}) =
\min\left(1, \frac{p(x \mid \mathbf{c})}{q(x \mid \mathbf{c})}\right).
\]

The **acceptance rate** for a draft token is

\[
\alpha =
\mathbb{E}_{x \sim q}\left[\beta(x)\right]
=
\sum_{x \in \mathcal{V}} \min\left(q(x), p(x)\right).
\]

Equivalently,

\[
\alpha = 1 - \mathrm{TV}(p, q),
\]

where

\[
\mathrm{TV}(p, q) = \frac{1}{2}\sum_{x \in \mathcal{V}} |p(x) - q(x)|.
\]

Thus maximizing acceptance rate is exactly equivalent to minimizing total variation distance:

\[
\max \alpha \iff \min \mathrm{TV}(p, q).
\]

The global optimum is \(q = p\), giving \(\alpha = 1\) and \(\mathrm{TV} = 0\). However, capacity-limited draft models usually cannot reach this optimum. At suboptimal points, minimizing KL divergence does not necessarily maximize acceptance rate.

## 2. Divergence Losses and Their Gradients

Common training objectives are:

\[
\mathrm{KL}(p \| q) = \sum_i p_i \log \frac{p_i}{q_i},
\]

\[
\mathrm{TV}(p, q) = \frac{1}{2}\sum_i |p_i - q_i|.
\]

The key relation is

\[
\alpha = 1 - \mathrm{TV}(p, q).
\]

### 2.1 Gradient of KL Divergence

With respect to draft logits \(z_q\),

\[
\boxed{
\nabla_{z_q} \mathrm{KL}(p \| q) = q - p
}
\]

Derivation sketch:

\[
\frac{\partial q_i}{\partial z_{q,j}} = q_i(\delta_{ij} - q_j)
\]

\[
\frac{\partial \mathrm{KL}}{\partial z_{q,j}}
=
-\sum_i \frac{p_i}{q_i} \frac{\partial q_i}{\partial z_{q,j}}
=
q_j - p_j.
\]

So KL gives a smooth gradient proportional to the probability error.

### 2.2 Gradient of Total Variation

Define

\[
s_i = \mathrm{sign}(q_i - p_i).
\]

Then

\[
\boxed{
\nabla_{z_q} \mathrm{TV}(p, q)
=
\frac{1}{2} q \odot \left(s - \mathbb{E}_q[s]\right)
}
\]

where

\[
\mathbb{E}_q[s] = \sum_i q_i s_i.
\]

Derivation sketch:

\[
\frac{\partial \mathrm{TV}}{\partial z_{q,j}}
=
\frac{1}{2}\sum_i s_i q_i(\delta_{ij} - q_j)
=
\frac{1}{2} q_j \left(s_j - \mathbb{E}_q[s]\right).
\]

### 2.3 Why Pure TV Is Difficult

TV directly targets acceptance rate, but pure TV optimization is problematic when training from random initialization:

1. The gradient depends only on the **sign** of the error, not its magnitude.
2. The loss is non-differentiable when \(q_i = p_i\).
3. For a randomly initialized draft with near-uniform \(q\) over a large vocabulary \(V\), the TV gradient norm is very small:

\[
\|\nabla_{z_q} \mathrm{TV}\| = \mathcal{O}\left(\frac{\sqrt{k}}{V}\right),
\]

where \(k\) is the number of tokens with non-negligible target probability.

KL, by contrast, has gradient norm

\[
\|\nabla_{z_q} \mathrm{KL}\| = \mathcal{O}\left(\frac{1}{\sqrt{k}}\right),
\]

which is much better conditioned early in training.

## 3. LK Losses

Two LK loss variants are proposed.

## 3.1 Hybrid LK Loss: \(\mathcal{L}_{\mathrm{LK}}^{\lambda}\)

\[
\boxed{
\mathcal{L}_{\mathrm{LK}}^{\lambda}(p,q)
=
\lambda \cdot \mathrm{KL}(p \| q)
+
(1-\lambda) \cdot \mathrm{TV}(p,q)
}
\]

Special cases:

- \(\lambda = 1\): standard KL training.
- \(\lambda = 0\): pure TV optimization.

### Adaptive schedule for \(\lambda\)

\[
\boxed{
\lambda = \exp\left(-\eta \cdot \mathrm{sg}[\alpha]\right),
\qquad \eta > 0
}
\]

where \(\mathrm{sg}[\cdot]\) is the stop-gradient operator.

Properties:

- If \(\alpha \to 0\), then \(\lambda \to 1\): KL dominates.
- If \(\alpha \to 1\), then \(\lambda \to 0\): TV dominates.
- The schedule smoothly transitions from KL-guided optimization to direct acceptance-rate optimization.

In practice, \(\lambda\) is computed **independently for each draft token position** using aggregated values of \(\alpha\) across sequence and batch dimensions.

Interpretation as a trust-region method:

\[
\min_q \mathrm{TV}(p,q)
\quad \text{s.t.} \quad
\mathrm{KL}(p \| q) \le \delta.
\]

Early on, KL keeps \(q\) close enough to \(p\) for TV gradients to be useful. Later, TV directly improves acceptance.

Default value from the paper: \(\eta = 3\).

## 3.2 Likelihood-Based LK Loss: \(\mathcal{L}_{\mathrm{LK}}^{\alpha}\)

\[
\boxed{
\mathcal{L}_{\mathrm{LK}}^{\alpha}(p,q)
=
-\log \alpha
=
-\log \sum_{x \in \mathcal{V}} \min\left(p(x), q(x)\right)
}
\]

This directly maximizes the marginal acceptance probability.

### Gradient

\[
\boxed{
\nabla_{z_q} \mathcal{L}_{\mathrm{LK}}^{\alpha}
=
\frac{1}{\alpha}
\nabla_{z_q} \mathrm{TV}(p,q)
}
\]

This means \(\mathcal{L}_{\mathrm{LK}}^{\alpha}\) performs TV optimization with adaptive gradient scaling. The factor \(1/\alpha\) amplifies gradients when acceptance is low, fixing the vanishing-gradient problem of pure TV.

Gradient magnitude in the early diffuse-\(q\) regime:

\[
\|\nabla_{z_q} \mathcal{L}_{\mathrm{LK}}^{\alpha}\|
=
\mathcal{O}\left(\frac{1}{\sqrt{k}}\right),
\]

matching KL’s early gradient scale while pointing in the TV direction.

### Degenerate case

If \(p\) is a point mass at \(x^*\), then

\[
\alpha = q(x^*),
\]

so

\[
\mathcal{L}_{\mathrm{LK}}^{\alpha}(p,q) = -\log q(x^*),
\]

which is the standard negative log-likelihood.

## 4. Vocabulary Truncation

Some draft architectures use a truncated draft vocabulary. Let \(m\) be a mask that sets target logits outside the draft vocabulary to \(-\infty\).

For KL training, one must avoid infinite KL by redefining the target distribution:

\[
\bar{p} = \mathrm{softmax}(m \odot z_p).
\]

Then KL is computed as

\[
\mathrm{KL}(\bar{p} \| q).
\]

This makes KL a proxy of a proxy.

LK losses handle truncation naturally. Tokens outside the draft vocabulary contribute

\[
\min(p_i, 0) = 0
\]

to the acceptance rate. Therefore, for \(\mathcal{L}_{\mathrm{LK}}^{\alpha}\) and the TV component of \(\mathcal{L}_{\mathrm{LK}}^{\lambda}\), no modification of the original target \(p\) is needed.

Important implementation note:

- For **KL component** in the hybrid loss, use renormalized target \(\bar{p}\) over the draft vocabulary.
- For **TV component** and **LK-alpha**, use the original target probabilities \(p\) over the full vocabulary, but gather only the draft-vocabulary entries. Tokens outside the draft vocabulary contribute zero to \(\alpha\).

## 5. Aggregation Across Draft Heads

Let \(K\) be the number of speculative heads. The \(n\)-th head receives weight

\[
\gamma^{n-1},
\qquad \gamma \in (0,1].
\]

The paper uses

\[
\gamma = 0.8.
\]

Total loss:

\[
\boxed{
\mathcal{L}_{\text{total}}
=
\sum_{n=1}^{K} \gamma^{n-1} \mathcal{L}_n
}
\]

where \(\mathcal{L}_n\) is the chosen LK loss at draft position \(n\).

## 6. Pseudocode for Implementation

```python
import torch
import torch.nn.functional as F

def compute_lk_loss(
    target_logits,        # [B, S, K, V_full] or [B, K, V_full]
    draft_logits,         # [B, S, K, V_draft] or [B, K, V_draft]
    draft_vocab_mask=None, # optional: indices of draft vocab in full vocab
    eta=3.0,
    gamma=0.8,
    eps=1e-8,
    mode="hybrid"         # "hybrid" or "lk_alpha"
):
    # Target and draft probabilities
    p_full = F.softmax(target_logits, dim=-1)
    q = F.softmax(draft_logits, dim=-1)

    if draft_vocab_mask is not None:
        # Original target probabilities on draft vocab. Do NOT renormalize for TV / alpha.
        p_draft = p_full[..., draft_vocab_mask]
        # Renormalized target for KL only.
        p_bar = F.softmax(target_logits[..., draft_vocab_mask], dim=-1)
    else:
        p_draft = p_full
        p_bar = p_full

    # Acceptance rate per draft position
    alpha = torch.minimum(p_draft, q).sum(dim=-1)  # shape [..., K]

    # Full TV with original target p. This is exact even with truncation
    # because tokens outside draft vocab contribute min(p, 0) = 0 to alpha.
    tv = 1.0 - alpha

    # KL against renormalized target for numerical stability / truncation
    kl = (p_bar * (torch.log(p_bar + eps) - torch.log(q + eps))).sum(dim=-1)

    if mode == "hybrid":
        # Aggregate alpha per head position over batch and sequence dimensions.
        # Example if alpha shape is [B, S, K]: mean over B and S.
        reduce_dims = tuple(range(alpha.ndim - 1))
        alpha_mean_per_head = alpha.mean(dim=reduce_dims)  # [K]
        lambda_per_head = torch.exp(-eta * alpha_mean_per_head.detach())  # [K]

        # Broadcast lambda to match loss shape
        lam = lambda_per_head.view(*([1] * (alpha.ndim - 1)), -1)
        loss_per_head = lam * kl + (1.0 - lam) * tv

    elif mode == "lk_alpha":
        loss_per_head = -torch.log(alpha + eps)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Exponential weighting across draft heads
    K = loss_per_head.shape[-1]
    weights = gamma ** torch.arange(K, device=loss_per_head.device)
    total_loss = (loss_per_head * weights).sum(dim=-1).mean()

    return total_loss
```

## 7. Summary of the Core Logic

1. Speculative decoding acceptance rate is exactly

\[
\alpha = \sum_x \min(p(x), q(x)) = 1 - \mathrm{TV}(p,q).
\]

2. KL is only a proxy. When draft capacity is limited, minimizing KL does not guarantee maximizing \(\alpha\).

3. Pure TV directly optimizes \(\alpha\), but its gradients vanish for randomly initialized drafts over large vocabularies.

4. Hybrid LK loss:

\[
\mathcal{L}_{\mathrm{LK}}^{\lambda}
=
\lambda \mathrm{KL}(p\|q) + (1-\lambda)\mathrm{TV}(p,q),
\]

with adaptive

\[
\lambda = \exp(-\eta \cdot \mathrm{sg}[\alpha]).
\]

5. Likelihood-based LK loss:

\[
\mathcal{L}_{\mathrm{LK}}^{\alpha} = -\log \alpha,
\]

whose gradient is

\[
\nabla_{z_q} \mathcal{L}_{\mathrm{LK}}^{\alpha}
=
\frac{1}{\alpha} \nabla_{z_q} \mathrm{TV}(p,q).
\]

6. Aggregate per-head losses with exponential decay \(\gamma^{n-1}\), typically \(\gamma=0.8\).

7. For truncated vocabularies: use original \(p\) for \(\alpha\) and TV; use renormalized \(\bar{p}\) only for the KL component.