from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
from transformer_lens.hook_points import HookPoint
from transformer_lens import HookedTransformer

from tlct.metrics.logit_diff import logit_diff
from tlct.utils.tl import get_cache, get_hook_name


@dataclass
class HeadRecoveryResult:
    baseline_clean: float
    baseline_corrupt: float
    head_patched: np.ndarray      # (n_layers, n_heads) patched logit diffs
    head_recovery: np.ndarray     # (n_layers, n_heads) normalized recovery


def _normalize_recovery(patched: np.ndarray, base_corrupt: float, base_clean: float, eps: float) -> np.ndarray:
    return (patched - base_corrupt) / (base_clean - base_corrupt + eps)


@torch.no_grad()
def compute_head_recovery(
    model: HookedTransformer,
    clean_prompt: str,
    corrupt_prompt: str,
    id_A: int,
    id_B: int,
    measure_pos: int = -1,
    patch_pos: int = -1,
    head_subset: Optional[List[List[int]]] = None,
    eps: float = 1e-9,
) -> HeadRecoveryResult:
    """
    Patches blocks.{layer}.attn.hook_result for a single head at patch_pos:
      act[:, patch_pos, head, :] <- clean_cache[...]

    Returns a full (layers x heads) matrix, unless head_subset is provided.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    baseline_clean = logit_diff(model, clean_prompt, id_A, id_B, pos=measure_pos)
    baseline_corrupt = logit_diff(model, corrupt_prompt, id_A, id_B, pos=measure_pos)

    clean_cache = get_cache(model, clean_prompt)

    head_patched = np.full((n_layers, n_heads), np.nan, dtype=np.float64)

    # which heads to evaluate
    if head_subset is None:
        pairs: List[Tuple[int, int]] = [(L, H) for L in range(n_layers) for H in range(n_heads)]
    else:
        pairs = [(int(L), int(H)) for L, H in head_subset]

    for layer, head in pairs:
        hook_name = get_hook_name(layer, "head_result")
        clean_act = clean_cache[hook_name]  # [batch, seq, head, d_head? or d_model/n_heads]

        def patch_one_head(act: torch.Tensor, hook: HookPoint) -> torch.Tensor:
            # act shape: [batch, seq, n_heads, d_head]
            act = act.clone()
            act[:, patch_pos, head, :] = clean_act[:, patch_pos, head, :]
            return act

        logits = model.run_with_hooks(
            corrupt_prompt,
            fwd_hooks=[(hook_name, patch_one_head)],
        )
        last = logits[0, measure_pos]
        head_patched[layer, head] = float(last[id_A] - last[id_B])

    head_recovery = _normalize_recovery(head_patched, baseline_corrupt, baseline_clean, eps)
    return HeadRecoveryResult(
        baseline_clean=baseline_clean,
        baseline_corrupt=baseline_corrupt,
        head_patched=head_patched,
        head_recovery=head_recovery,
    )
