"""
Compare per-depth MTP acceptance for a fixed raw prompt, spec_decode (torch)
vs the llama.cpp server.  Uses the same greedy decoding and prints
UNCONDITIONAL per-depth rates (accepted_depth / rounds with that depth reached
divided by all rounds), plus tokens/round.

Usage:
  python scripts/compare_accept.py "PROMPT" [max_new]
"""

# --- project-root bootstrap --------------------------------------------------
# Allow `python scripts/<name>.py` / `python tests/<name>.py` from anywhere:
# put the project root (for `import mtp`, `import src.model`) and the scripts/
# directory (for sibling tool imports) on sys.path.
import os as _os
import sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, _os.path.join(_ROOT, "scripts")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# -----------------------------------------------------------------------------
import sys, torch
from spec_decode import SpecDecoder, RWKV_TOKENIZER

PROMPT = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"
MAXNEW = int(sys.argv[2]) if len(sys.argv) > 2 else 128

dev = "cuda:0"
sd = SpecDecoder("rwkv7-g1k-1.5b-20260930-ctx25600.pth", "config-1.5b.json",
                 "mtp_out_ob_1.5b_lk/mtp_heads.pt", dev, arch="rwkv7")
tok = RWKV_TOKENIZER("rwkv_vocab_v20230424.txt")
ids = tok.encode(PROMPT)
print("prompt tokens:", ids)

greedy = True
gen = torch.Generator(device=dev).manual_seed(0)
idx = torch.tensor(ids, device=dev).unsqueeze(0)
hidden, _, state = sd.model.forward(idx, None, return_hidden=True)
h = hidden[:, -1, :]
sd.runner.prefill(hidden[0], list(ids))
out = list(ids)
start = len(out)
K = sd.num_mtp
acc = [0] * K
att = [0] * K
steps = 0
while len(out) - start < MAXNEW:
    steps += 1
    p = len(out) - 1
    snap = sd.runner.snapshot()
    pick = lambda zz: sd._pick(zz @ sd.head, greedy, 0, 0, 0, gen)
    drafts = sd.runner.draft_chain(h, p, pick)
    draft_t = torch.cat(drafts, dim=0)
    L = draft_t.numel()
    hidden_v, logits_v, state_v = sd.model.forward(draft_t.view(1, L), state, return_hidden=True)
    preds = sd._pick(logits_v[0], greedy, 0, 0, 0, gen)
    m = 0
    while m < L - 1 and preds[m].item() == draft_t[m + 1].item():
        m += 1
    for j in range(L - 1):
        if j <= m:
            att[j] += 1
            if j < m:
                acc[j] += 1
    emitted = draft_t[: m + 1].tolist() + [int(preds[m].item())]
    if m == L - 1:
        bonus = torch.tensor([[int(preds[m].item())]], device=dev)
        hidden2, _, state = sd.model.forward(bonus, state_v, return_hidden=True)
    else:
        commit = torch.tensor([emitted], device=dev)
        hidden2, _, state = sd.model.forward(commit, state, return_hidden=True)
    for tk in emitted:
        out.append(tk)
    if m == L - 1:
        H_new = torch.cat([hidden_v[0], hidden2[:, -1, :]], dim=0)
    else:
        H_new = hidden2[0]
    sd.runner.restore(snap)
    sd.runner.advance(h, H_new, p, out)
    h = hidden2[:, -1, :]
    if 0 in emitted:
        break

print(f"spec: rounds={steps} tokens={len(out)-start} tokens/round={ (len(out)-start)/max(steps,1):.2f}")
print("spec unconditional acc/pos:", [round(a / max(steps, 1), 3) for a in acc])
print("spec conditional   acc/att:", [round(a / max(b, 1), 3) for a, b in zip(acc, att)])
print("spec text:", tok.decode(out[len(ids):][:80]).replace("\n", " "))
