"""
Checkpoint I/O and end-to-end model construction for the MTP retrofit.

* :func:`make_args_from_config` / :func:`load_pth_state_dict` -- read the frozen
  backbone's config and monolithic ``.pth``;
* :func:`build_frozen_model` -- construct ``RWKVWithMTP``, resize the
  low-rank params to the checkpoint shapes, load, and freeze the backbone;
* :func:`save_mtp_heads` / :func:`load_mtp_heads` / :func:`peek_mtp_arch` --
  the arch-tagged ``mtp_heads.pt`` payload.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .backbone import _make_rwkv_with_mtp
from .heads import MTP_ARCHS


def make_args_from_config(config_path: str, ctx_len: int = 16384) -> argparse.Namespace:
    """Build the minimal args namespace ``src.model.RWKV`` needs."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    n_embd = cfg["hidden_size"]
    args = argparse.Namespace()
    args.n_embd = n_embd
    args.n_layer = cfg["num_hidden_layers"]
    args.vocab_size = cfg["vocab_size"]
    args.dim_att = n_embd
    # intermediate_size is authoritative; RWKV.__init__ would otherwise default
    # to 3.5*n_embd (8960 != 10240 for this model).
    args.dim_ffn = cfg.get("intermediate_size", int((n_embd * 3.5) // 32 * 32))
    args.head_size = cfg.get("head_dim", 64)
    args.ctx_len = ctx_len
    args.my_testing = "x070"
    args.grad_cp = 0
    # Only used by RWKV.configure_optimizers(), which this retrofit never calls.
    args.weight_decay = 0.0
    args.lr_init = 1e-5
    args.betas = (0.9, 0.99)
    args.adam_eps = 1e-18
    args.accelerator = "gpu"
    args.dtype = cfg.get("dtype", "bfloat16")
    args.pad_token_id = cfg.get("pad_token_id", 0)
    return args


_PREFIXES = ("_forward_module.", "module.", "rwkv7.", "rwkv.")


def load_pth_state_dict(pth_path: str) -> Dict[str, torch.Tensor]:
    """Load a forward .pth checkpoint, unwrap wrappers and strip known prefixes.

    Handles the common cases:
      * raw ``state_dict`` (this repo's checkpoint)
      * ``{"model": sd}`` / ``{"module": sd}`` / ``{"state_dict": sd}``
      * ``rwkv.`` / ``rwkv7.`` / ``module.`` / ``_forward_module.`` key prefixes
    """
    ckpt = torch.load(pth_path, map_location="cpu", weights_only=True, mmap=True)

    if isinstance(ckpt, dict):
        vals = list(ckpt.values())
        looks_like_sd = all(isinstance(v, torch.Tensor) for v in vals[:8])
        if not looks_like_sd:
            for wk in ("model", "module", "state_dict", "model_state_dict"):
                sub = ckpt.get(wk)
                if isinstance(sub, dict):
                    ckpt = sub
                    break
            else:
                raise RuntimeError(
                    f"Checkpoint {pth_path} is not a state_dict and no known "
                    f"wrapper key was found. Top-level keys: {list(ckpt.keys())[:20]}"
                )

    sd: Dict[str, torch.Tensor] = {}
    for k, v in ckpt.items():
        nk = k
        for p in _PREFIXES:
            if nk.startswith(p):
                nk = nk[len(p):]
                break
        sd[nk] = v
    return sd


def _resize_params_to_checkpoint(model: nn.Module,
                                 sd: Dict[str, torch.Tensor]) -> int:
    """Replace params whose shape differs from the checkpoint, in place.

    The G1 checkpoint uses explicit low-rank dims from config.json
    (a/decay=96, v=64, gate=320) while src/model.py derives them from heuristic
    formulas (128/128/96/256 for C=2560).  ``load_state_dict`` rejects shape
    mismatches unconditionally (it ignores ``assign=True`` for this check), so we
    rebuild those parameter objects at the checkpoint shapes first.  The tmix
    forward is shape-agnostic in these dims, so this is equivalent to having
    constructed the module with the config dims.
    """
    changed = 0
    msd = model.state_dict()
    for key, tensor in sd.items():
        cur = msd.get(key)
        if cur is None or cur.shape == tensor.shape:
            continue
        parts = key.split(".")
        parent = model
        for p in parts[:-1]:
            parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
        attr = parts[-1]
        old = getattr(parent, attr)
        if isinstance(old, nn.Parameter):
            # Inherit the current param's device/dtype: the sd may live on CPU
            # (torch.load(map_location='cpu')) while the module is already on
            # GPU (load_mtp_heads resizes after build_frozen_model moved it).
            new = nn.Parameter(torch.empty_like(tensor).to(device=old.device,
                                                           dtype=old.dtype),
                               requires_grad=old.requires_grad)
        else:
            new = torch.empty_like(tensor).to(device=old.device, dtype=old.dtype)
        setattr(parent, attr, new)
        changed += 1
    return changed


def build_frozen_model(
    pth_path: str,
    config_path: str,
    num_mtp: int = 3,
    ctx_len: int = 16384,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    shared_head: bool = False,
    arch: str = "mlp",
    eagle3_fusion: bool = False,
    warm_start_heads: bool = True,
):
    """Construct ``RWKVWithMTP`` from a monolithic .pth, freeze the backbone.

    ``shared_head=True`` builds the FastMTP-style variant: one MTP head module
    applied ``num_mtp`` times recursively.  It is checkpoint-compatible with a
    non-shared ``num_mtp=1`` model (same ``mtp_heads.0.*`` keys), so a
    single-step head can be fine-tuned into a shared recursive one.

    ``arch`` selects the head module: ``mlp`` (position-local, the original)
    or ``rwkv7`` (one full RWKV-7 block; carries a recurrent state at
    inference).  The arch is stored in checkpoints by ``save_mtp_heads`` and
    verified by ``load_mtp_heads``.

    ``eagle3_fusion`` (rwkv7 head only) builds the EAGLE-3 multi-layer fusion
    parameters (``ln_early``/``ln_mid``/``fc_fuse``) plus per-depth input
    LayerNorms inside each head module; they live under ``mtp_heads.*`` so
    checkpointing is unchanged.

    ``warm_start_heads`` (rwkv7 head only) copies the pretrained weights of
    ``model.blocks[-1]`` and ``model.ln_out`` into every head module after the
    backbone load, so a fresh head starts as a functional copy of the last
    backbone layer instead of a random block.  Harmless when a trained head is
    loaded on top (``load_mtp_heads`` overwrites the head weights anyway), so
    callers only disable it to reproduce the old random-init exactly.
    """
    if arch not in MTP_ARCHS:
        raise ValueError(f"unknown --mtp-arch {arch!r}; choose from {MTP_ARCHS}")
    # Defensive: pin the process to the target GPU before anything CUDA runs.
    # The vendor .cu kernels launch on the *current* device (getCurrentCUDAStream()
    # / bare <<<>>> launches, no device guard), so building with device="cuda:1"
    # while the process current device is still cuda:0 runs every custom kernel
    # on the wrong GPU against cuda:1 pointers; the corruption later surfaces as
    # bogus cublasCreate/GemmEx errors.  Idempotent when the caller already
    # pinned the device (train_mtp.py does).
    if isinstance(device, str) and ":" in device:
        torch.cuda.set_device(device)
    RWKVWithMTP = _make_rwkv_with_mtp()
    args = make_args_from_config(config_path, ctx_len)
    sd = load_pth_state_dict(pth_path)

    # Note: construct in fp32 (nn.init.orthogonal_ has no bf16 CPU kernel), then
    # resize + cast.  The transient fp32 backbone is ~12 GB RAM; it is freed once
    # the model is cast to bf16 below.
    model = RWKVWithMTP(args, num_mtp, shared_head=shared_head, arch=arch,
                        eagle3_fusion=eagle3_fusion)

    n_resized = _resize_params_to_checkpoint(model, sd)
    if n_resized:
        print(f"[mtp] resized {n_resized} low-rank params to checkpoint shapes")

    model.to(dtype=dtype)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    del sd

    bad_missing = [k for k in missing if not k.startswith("mtp_heads.")]
    if bad_missing:
        raise RuntimeError(
            f"{len(bad_missing)} backbone keys missing from checkpoint, e.g. "
            f"{bad_missing[:20]}"
        )
    if unexpected:
        print(f"[mtp] WARNING: {len(unexpected)} unexpected checkpoint keys, "
              f"e.g. {unexpected[:10]}")

    # Backbone tensors are already bf16 from the checkpoint; only the freshly
    # initialized MTP heads still need casting.
    model.mtp_heads.to(dtype)

    if arch == "rwkv7" and warm_start_heads:
        # EAGLE-3-style warm start: the draft head's block begins as an exact
        # copy of the backbone's last block (and the head's final LayerNorm as
        # a copy of ln_out), so step-0 drafts come from a competent function
        # instead of a random init.  Combined with the [0, 0, I] fc_fuse init,
        # the whole depth-0 input path is a no-op-reparameterized copy of the
        # backbone's own last-layer computation at step 0.
        with torch.no_grad():
            last_blk = model.blocks[-1]
            for mtp_head in model.mtp_heads:
                # The head block was built with src.model's heuristic low-rank
                # dims (a/decay=128, v=128, gate=224 at C=2048) while the
                # checkpoint uses the config dims (96/64/256) -- resize to the
                # checkpoint shapes first, exactly like the backbone load did.
                _resize_params_to_checkpoint(mtp_head.block, last_blk.state_dict())
                mtp_head.block.ln1.load_state_dict(last_blk.ln1.state_dict())
                mtp_head.block.ln2.load_state_dict(last_blk.ln2.state_dict())
                mtp_head.block.att.load_state_dict(last_blk.att.state_dict(), strict=False)
                mtp_head.block.ffn.load_state_dict(last_blk.ffn.state_dict())
                # Re-alias dead layer-0 value-residual params to a0/a1/a2 (a
                # load_state_dict into att rebinds the aliased parameters).
                mtp_head.block.att.v0 = mtp_head.block.att.a0
                mtp_head.block.att.v1 = mtp_head.block.att.a1
                mtp_head.block.att.v2 = mtp_head.block.att.a2
                if hasattr(mtp_head, "ln_out"):
                    mtp_head.ln_out.load_state_dict(model.ln_out.state_dict())
        print(f"[mtp] warm-started {len(model.mtp_heads)} rwkv7 head block(s) "
              f"from blocks[-1] + ln_out")

    for name, p in model.named_parameters():
        p.requires_grad_("mtp_heads" in name)

    model.freeze_backbone = True
    model.to(device=device)
    model.train()
    return model, args


def mtp_parameters(model) -> List[nn.Parameter]:
    return [p for n, p in model.named_parameters() if "mtp_heads" in n]


def count_mtp_params(model) -> int:
    return sum(p.numel() for p in mtp_parameters(model))


def save_mtp_heads(model, path: str, meta: Optional[dict] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    eagle3 = bool(getattr(getattr(model, "mtp_heads", [None])[0],
                          "eagle3_fusion", False))
    payload = {
        "mtp_heads": model.mtp_heads.state_dict(),
        "num_mtp": int(model.num_mtp),
        "n_embd": int(model.args.n_embd),
        "shared_head": bool(getattr(model, "shared_head", False)),
        "arch": str(getattr(model, "mtp_arch", "mlp")),
        "eagle3_fusion": eagle3,
        "meta": meta or {},
    }
    torch.save(payload, path)


def _payload_arch(payload: dict) -> str:
    """Arch tag of an mtp_heads.pt payload ('mlp' for legacy files)."""
    if not isinstance(payload, dict):
        return "mlp"
    arch = payload.get("arch") or (payload.get("meta") or {}).get("arch")
    return str(arch) if arch else "mlp"


def peek_mtp_arch(path: str) -> str:
    """Read just the arch tag from an ``mtp_heads.pt`` without building a model.

    Lets consumers (eval_mtp / spec_decode / train --load-mtp) auto-detect the
    head architecture; legacy checkpoints without a tag read as ``mlp``.
    """
    return str(peek_mtp_meta(path).get("arch", "mlp"))


def peek_mtp_meta(path: str) -> dict:
    """Read the structural tags (arch, shared_head, eagle3_fusion, num_mtp)
    from an ``mtp_heads.pt`` without building a model.  Legacy checkpoints
    without tags return the defaults (arch 'mlp', flags off)."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    meta = dict(payload.get("meta", {}) or {}) if isinstance(payload, dict) else {}
    out = {
        "arch": str(payload.get("arch") or meta.get("arch") or "mlp"),
        "shared_head": bool(payload.get("shared_head", meta.get("shared_head", False))),
        "eagle3_fusion": bool(payload.get("eagle3_fusion",
                                          meta.get("eagle3_fusion", False))),
    }
    if payload.get("num_mtp") is not None:
        out["num_mtp"] = int(payload["num_mtp"])
    return out


def load_mtp_heads(model, path: str, strict: bool = True) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    ckpt_arch = _payload_arch(payload)
    model_arch = str(getattr(model, "mtp_arch", "mlp"))
    if ckpt_arch != model_arch:
        raise RuntimeError(
            f"mtp arch mismatch: checkpoint {path} was trained with "
            f"--mtp-arch {ckpt_arch} but the model was built with "
            f"--mtp-arch {model_arch}; rebuild with the matching --mtp-arch"
        )
    sd = payload["mtp_heads"] if "mtp_heads" in payload else payload
    ckpt_eagle3 = bool(payload.get("eagle3_fusion",
                                   (payload.get("meta") or {}).get("eagle3_fusion",
                                                                    False)))
    model_eagle3 = bool(getattr(getattr(model, "mtp_heads", [None])[0],
                                "eagle3_fusion", False))
    if ckpt_eagle3 != model_eagle3:
        raise RuntimeError(
            f"eagle3_fusion mismatch: checkpoint {path} was trained with "
            f"eagle3_fusion={ckpt_eagle3} but the model was built with "
            f"eagle3_fusion={model_eagle3}; rebuild with the matching "
            f"--eagle3-fusion setting")
    # EAGLE-3-era checkpoints store the head block at the BACKBONE's low-rank
    # dims (the warm-start copied blocks[-1] after its own resize), while a
    # freshly built head uses src.model's heuristic dims -- resize to the
    # payload shapes first (same discipline as the backbone load).  This also
    # rebinds att.v1/v2 onto fresh params, breaking the layer-0 v0->a0 alias,
    # so restore the alias afterwards (values are equal in the payload anyway;
    # layer 0 never takes the value-residual branch).
    _resize_params_to_checkpoint(model.mtp_heads, sd)
    for mtp_head in model.mtp_heads:
        att = getattr(mtp_head, "block", None) and getattr(mtp_head.block, "att", None)
        if att is not None and all(hasattr(att, a) for a in ("v0", "a0",
                                                             "v1", "a1",
                                                             "v2", "a2")):
            att.v0 = att.a0
            att.v1 = att.a1
            att.v2 = att.a2
    missing, unexpected = model.mtp_heads.load_state_dict(sd, strict=False)
    if strict and missing and not unexpected and not ckpt_eagle3:
        legacy_missing = [k for k in missing
                          if k.endswith("ln_out.weight") or k.endswith("ln_out.bias")
                          or ".ln_h_extra." in k]
        if legacy_missing and len(legacy_missing) == len(missing):
            # Legacy checkpoint predating the EAGLE-3 head changes.  Its draft
            # z was UNnormalized and EVERY shared-head depth normalized its
            # input with the SAME ln_h affine, so the faithful compatibility
            # init is: identity ln_out (weight 1, bias 0 == no-op) and
            # ln_h_extra[k-1] := ln_h -- zero-shock for both resumed training
            # and evaluation.  Fresh eagle3 runs warm-start ln_out from the
            # backbone and train their own ln_h_extra instead.
            with torch.no_grad():
                for mtp_head in model.mtp_heads:
                    if hasattr(mtp_head, "ln_out"):
                        nn.init.ones_(mtp_head.ln_out.weight)
                        nn.init.zeros_(mtp_head.ln_out.bias)
                    for i, ln_extra in enumerate(getattr(mtp_head, "ln_h_extra", [])):
                        ln_extra.weight.copy_(mtp_head.ln_h.weight)
                        ln_extra.bias.copy_(mtp_head.ln_h.bias)
            print(f"[mtp] legacy checkpoint (pre-EAGLE-3 head): initialized "
                  f"ln_out to identity and ln_h_extra[k] := ln_h")
            # The legacy payload genuinely lacks these keys (it predates them);
            # the compat init above supplies their values, so finish the load
            # non-strict and make sure nothing ELSE is missing.
            missing, unexpected = model.mtp_heads.load_state_dict(sd, strict=False)
            missing = [k for k in missing if k not in set(legacy_missing)]
    if strict and (missing or unexpected):
        raise RuntimeError(f"load_mtp_heads mismatch: missing={missing}, unexpected={unexpected}")
    if int(payload.get("num_mtp", model.num_mtp)) != model.num_mtp:
        print(f"[mtp] WARNING: checkpoint num_mtp={payload.get('num_mtp')} "
              f"!= model.num_mtp={model.num_mtp}")
    # Expose the structural flags so callers (eval/spec scripts) can verify the
    # model they built matches the checkpoint without re-reading the payload.
    meta = dict(payload.get("meta", {}) or {})
    meta.setdefault("num_mtp", int(payload.get("num_mtp", model.num_mtp)))
    meta.setdefault("shared_head", bool(payload.get("shared_head", False)))
    meta.setdefault("arch", ckpt_arch)
    meta.setdefault("eagle3_fusion", ckpt_eagle3)
    return meta
