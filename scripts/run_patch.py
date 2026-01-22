from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from tlct.utils.io import load_yaml, ensure_dir, save_json, save_npz
from tlct.utils.seed import set_seed
from tlct.utils.tl import load_model
from tlct.experiments.head_patching import compute_head_recovery


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/default.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    set_seed(int(cfg["experiment"]["seed"]))

    model = load_model(
        cfg["model"]["name"],
        cfg["model"]["device"],
        cfg["model"]["dtype"],
    )

    # Run experiment
    res = compute_head_recovery(
        model=model,
        clean_prompt=cfg["task"]["clean_prompt"],
        corrupt_prompt=cfg["task"]["corrupt_prompt"],
        id_A=int(cfg["task"]["id_A"]),
        id_B=int(cfg["task"]["id_B"]),
        measure_pos=int(cfg["task"].get("measure_pos", -1)),
        patch_pos=int(cfg["patching"].get("patch_pos", -1)),
        head_subset=cfg["patching"].get("head_subset", None),
        eps=float(cfg["patching"].get("eps", 1e-9)),
    )

    # Save
    out_dir = ensure_dir(cfg["experiment"]["output_dir"])
    tag = cfg["experiment"].get("run_tag", "run")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{cfg['experiment']['name']}_{tag}_{ts}"

    npz_path = out_dir / f"{stem}.npz"
    json_path = out_dir / f"{stem}.json"

    save_npz(
        npz_path,
        head_patched=res.head_patched,
        head_recovery=res.head_recovery,
        baseline_clean=np.array([res.baseline_clean]),
        baseline_corrupt=np.array([res.baseline_corrupt]),
    )
    save_json(json_path, cfg)

    # Print TopK
    topk = int(cfg["analysis"].get("topk", 10))
    flat = np.argsort(-np.nan_to_num(res.head_recovery, nan=-1.0).ravel())[:topk]
    n_heads = model.cfg.n_heads
    print("Top heads by recovery:")
    for idx in flat:
        layer = idx // n_heads
        head = idx % n_heads
        r = res.head_recovery[layer, head]
        print(f"Layer {layer:02d} Head {head:02d} | recovery={r:.3f}")

    print(f"\nSaved: {npz_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
