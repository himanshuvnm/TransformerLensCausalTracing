from __future__ import annotations
from typing import Dict, Any, Tuple

import torch
from transformer_lens import HookedTransformer


def load_model(model_name: str, device: str, dtype: str) -> HookedTransformer:
    # dtype string -> torch dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.float16)

    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=torch_dtype,
    )
    return model


def get_cache(model: HookedTransformer, prompt: str) -> Dict[str, torch.Tensor]:
    _, cache = model.run_with_cache(prompt)
    return cache


def get_hook_name(layer: int, kind: str) -> str:
    """
    kind:
      - "head_result" -> blocks.{layer}.attn.hook_result
      - "resid_post"  -> blocks.{layer}.hook_resid_post
      - "mlp_out"     -> blocks.{layer}.hook_mlp_out
    """
    if kind == "head_result":
        return f"blocks.{layer}.attn.hook_result"
    if kind == "resid_post":
        return f"blocks.{layer}.hook_resid_post"
    if kind == "mlp_out":
        return f"blocks.{layer}.hook_mlp_out"
    raise ValueError(f"Unknown kind: {kind}")
