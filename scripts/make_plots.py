from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from tlct.utils.io import load_npz, load_yaml, ensure_dir
from tlct.plots.head_recovery import plot_head_recovery


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_npz", type=str, required=True, help="Path to results .npz")
    ap.add_argument("--config", type=str, default="config/default.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    data = load_npz(args.run_npz)
    head_recovery = data["head_recovery"]

    n_layers, n_heads = head_recovery.shape
    topk = int(cfg["analysis"].get("topk", 10))
    flat_idx = np.argsort(-np.nan_to_num(head_recovery, nan=-1.0).ravel())[:topk]

    clip = cfg["plot"].get("heatmap_percentile_clip", [5, 95])
    fig = plot_head_recovery(
        head_recovery=head_recovery,
        flat_idx=flat_idx,
        n_layers=n_layers,
        n_heads=n_heads,
        topk_annotate=int(cfg["plot"].get("annotate_topk", 10)),
        clip_percentiles=(float(clip[0]), float(clip[1])),
    )

    out_path = Path(cfg["plot"].get("out_path", "figures/head_recovery.png"))
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()
