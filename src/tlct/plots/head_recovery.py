from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


def plot_head_recovery(
    head_recovery: np.ndarray,
    flat_idx: np.ndarray,
    n_layers: int,
    n_heads: int,
    topk_annotate: int = 10,
    clip_percentiles: Tuple[float, float] = (5, 95),
    figsize: Tuple[int, int] = (16, 5),
):
    flat_idx = np.asarray(flat_idx, dtype=int)
    layers = flat_idx // n_heads
    heads = flat_idx % n_heads
    vals = head_recovery[layers, heads]

    order = np.argsort(-vals)
    layers, heads, vals = layers[order], heads[order], vals[order]

    topk = min(topk_annotate, len(vals))
    top_layers, top_heads, top_vals = layers[:topk], heads[:topk], vals[:topk]

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 1.4])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # Left: recovery vs head colored by layer
    sc1 = ax1.scatter(heads, vals, c=layers, cmap="viridis", s=35, alpha=0.75, edgecolors="none")
    ax1.set_title("Selected heads: recovery vs head (colored by layer)")
    ax1.set_xlabel("Head")
    ax1.set_ylabel("Recovery")
    ax1.set_xlim(-0.5, n_heads - 0.5)
    ax1.grid(True, alpha=0.25)
    cb1 = fig.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04)
    cb1.set_label("Layer")

    for L, H, R in zip(top_layers, top_heads, top_vals):
        ax1.scatter([H], [R], s=90, facecolors="none", edgecolors="k", linewidths=1.2)
        ax1.annotate(f"L{L}:H{H}\n{R:.3f}", (H, R), textcoords="offset points", xytext=(6, 6), fontsize=8)

    # Middle: recovery vs layer colored by head
    sc2 = ax2.scatter(layers, vals, c=heads, cmap="plasma", s=35, alpha=0.75, edgecolors="none")
    ax2.set_title("Selected heads: recovery vs layer (colored by head)")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Recovery")
    ax2.set_xlim(-0.5, n_layers - 0.5)
    ax2.grid(True, alpha=0.25)
    cb2 = fig.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.set_label("Head")

    for L, H, R in zip(top_layers, top_heads, top_vals):
        ax2.scatter([L], [R], s=90, facecolors="none", edgecolors="k", linewidths=1.2)
        ax2.annotate(f"H{H}\n{R:.3f}", (L, R), textcoords="offset points", xytext=(6, 6), fontsize=8)

    # Right: heatmap
    vmin = np.nanpercentile(head_recovery, clip_percentiles[0])
    vmax = np.nanpercentile(head_recovery, clip_percentiles[1])
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    im = ax3.imshow(head_recovery, aspect="auto", origin="lower", norm=norm)
    ax3.set_title("Head recovery heatmap (all layers/heads)")
    ax3.set_xlabel("Head")
    ax3.set_ylabel("Layer")

    # overlay selected + top
    ax3.scatter(heads, layers, s=18, facecolors="none", edgecolors="w", linewidths=0.8, alpha=0.9)
    ax3.scatter(top_heads, top_layers, s=80, facecolors="none", edgecolors="k", linewidths=1.5)

    cb3 = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cb3.set_label("Recovery")
    return fig
