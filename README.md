# Transformer Lens Causal Tracing

**TransformerLens Causal Tracing** is a reproducible research codebase for performing **activation patching / causal tracing** experiments on transformer models using the [TransformerLens](https://github.com/neelnanda-io/TransformerLens) library.

This repository provides:
- Config-driven experiments to locate causal components (e.g., heads)
- Utilities for computing logit differences
- Scripts for running large sweeping experiments
- Plotting of head-level recovery heatmaps
- Reproducible configs and command-line scripts

---

## 🔍 What is Causal Tracing?

Causal tracing (sometimes called activation patching) is a mechanistic interpretability technique that:
1. Runs a *clean* and *corrupt* version of a prompt
2. At each layer or component (head/MLP), replaces activations in the corrupt run with those from the clean run
3. Measures how much the final prediction recovers toward the clean run

By comparing the recovery per component, researchers can identify **which parts of the network are causally important** for a given behavior.

---

## 🚀 Features

✔ Configurable experiments via YAML  
✔ Supports TransformerLens models (e.g., `gpt2-small`)  
✔ Head-level recovery computation  
✔ Normalized recovery scores  
✔ Beautiful plots (scatter + heatmap)  
✔ Reproducible run artifacts (`.npz`, `.json`, figures)

## 📊 Plotting Result

<img width="18098" height="6116" alt="image" src="https://github.com/user-attachments/assets/c6b07f02-1234-453f-b202-23f303ac8919" />
