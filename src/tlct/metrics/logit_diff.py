from __future__ import annotations
from transformer_lens import HookedTransformer


def logit_diff(model: HookedTransformer, prompt: str, id_A: int, id_B: int, pos: int = -1) -> float:
    logits = model(prompt)  # [batch, seq, vocab]
    last = logits[0, pos]
    return float(last[id_A] - last[id_B])
