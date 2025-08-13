#!/usr/bin/env python3
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= EDIT THESE =========
CSV_PATH = "/home/glene/luna/CPOM/glene/PostDoc/5DAIS/Methods/ML/GEE/csv/metrics_fullscene.csv"
OUT_DIR  = Path("/home/glene/luna/CPOM/glene/PostDoc/5DAIS/Methods/ML/comparison_figures")

# GEE class ids: 0=background, 1=lake, 2=slush
CLASS_IDS   = [0, 1, 2]
CLASS_NAMES = {0: "background", 1: "lake", 2: "slush"}
# =============================

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Clean minimalist style (no grids)
plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": False,
    "axes.spines.top": False,
    "legend.frameon": False,
})

# ----- load + coerce numerics -----
df = pd.read_csv(CSV_PATH)
for c in df.columns:
    if c not in ("Scene", "Pair"):
        df[c] = pd.to_numeric(df[c], errors="ignore")

scenes = df["Scene"].astype(str).tolist()

# ---------- helpers ----------
def extract_stratified_cm(row):
    """Return stratified 3x3 CM (np.array) if CM_strat_* columns exist; else None."""
    needed = [f"CM_strat_t{t}_p{p}" for t in CLASS_IDS for p in CLASS_IDS]
    if not all(col in row for col in needed):
        return None
    m = np.zeros((len(CLASS_IDS), len(CLASS_IDS)), dtype=int)
    for i, t in enumerate(CLASS_IDS):
        for j, p in enumerate(CLASS_IDS):
            m[i, j] = int(row[f"CM_strat_t{t}_p{p}"])
    return m

# ======================================================================
# 1) ALL STRATIFIED CONFUSION MATRICES IN ONE FIGURE (no colorbar)
# ======================================================================
cms, cm_scenes = [], []
for _, row in df.iterrows():
    cm = extract_stratified_cm(row)
    if cm is None:
        print(f"[WARN] {row.get('Scene')}: no CM_strat_* columns found; skipping in CM grid.")
        continue
    cms.append(cm)
    cm_scenes.append(str(row.get("Scene")))

if cms:
    n = len(cms)
    ncols = 3
    nrows = 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3.2, nrows*3.2))
    axes = np.atleast_2d(axes)

    for idx, (scene, cm) in enumerate(zip(cm_scenes, cms)):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        ax.imshow(cm, cmap="Blues")  # no colorbar
        ax.set_xticks(np.arange(len(CLASS_IDS)))
        ax.set_yticks(np.arange(len(CLASS_IDS)))
        ax.set_xticklabels([CLASS_NAMES[i] for i in CLASS_IDS], rotation=25, ha='right', fontsize=8)
        ax.set_yticklabels([CLASS_NAMES[i] for i in CLASS_IDS], fontsize=8)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("True", fontsize=8)
        ax.set_title(scene, fontsize=9)

        # annotate counts
        vmax = cm.max()
        thresh = vmax / 2.0 if vmax > 0 else 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        fontsize=8,
                        color="white" if cm[i, j] > thresh else "black")

    # hide unused axes
    for k in range(n, nrows*ncols):
        r, c = divmod(k, ncols)
        axes[r, c].axis("off")

    cm_grid_path = OUT_DIR / "all_confusion_matrices_stratified.png"
    fig.tight_layout()
    fig.savefig(cm_grid_path)
    plt.close(fig)
    print("Saved:", cm_grid_path)
else:
    print("[INFO] No stratified CM grid produced.")

# ======================================================================
# 2) PER-CLASS PRECISION / RECALL / F1 / IoU — blue-hued grouped bars
# ======================================================================
metrics_by_prefix = [("Precision", "Prec_class"),
                     ("Recall",    "Rec_class"),
                     ("F1",        "F1_class"),
                     ("IoU",       "IoU_class")]

# blue hue palette (darker to lighter)
blue_palette = ["#08306B", "#2171B5", "#6BAED6", "#C6DBEF"]

x = np.arange(len(scenes)) * 1.1
bar_width = 0.18

fig, axes = plt.subplots(3, 1, figsize=(max(11, len(scenes)*0.7), 9), sharex=True)

for rowi, cls in enumerate(CLASS_IDS):
    ax = axes[rowi]
    offset = -1.5 * bar_width  # center the 4 bars per group
    for color, (label, pref) in zip(blue_palette, metrics_by_prefix):
        col = f"{pref}{cls}"
        if col not in df.columns:
            continue
        y = pd.to_numeric(df[col], errors="coerce").astype(float).values
        ax.bar(x + offset, y,
               width=bar_width,
               label=label,
               color=color,
               alpha=0.95,
               edgecolor="black",
               linewidth=0.3)
        offset += bar_width

    ax.set_ylim(0.75, 1.0)
    ax.set_ylabel(CLASS_NAMES[cls].capitalize())
    if rowi == 0:
        ax.set_title("Per-class Precision / Recall / F1 / IoU across ice shelves", pad=6)
        ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.25), frameon=False)

axes[-1].set_xticks(x)
axes[-1].set_xticklabels(scenes, rotation=60, ha='right')
axes[-1].set_xlabel("Ice shelf (Scene)")

metrics_path = OUT_DIR / "per_class_metrics_bars_blue.png"
fig.tight_layout()
fig.savefig(metrics_path)
plt.close(fig)
print("Saved:", metrics_path)


# ======================================================================
# 3) TRIPLE-PANEL ERROR CHART (dual y-axis)
# ======================================================================
panel_order = [0, 1, 2]  # top=background (0), middle=lake (1), bottom=slush (2)

relerr = {c: [] for c in CLASS_IDS}
abserr = {c: [] for c in CLASS_IDS}

# Extract shortened scene names for x-axis
short_scenes = [str(s).split("_")[0] for s in scenes]

for _, row in df.iterrows():
    for c in CLASS_IDS:
        rel_col = f"RelErr_class{c}"
        dlt_col = f"Delta_px_class{c}"
        r = abs(float(row[rel_col])) if rel_col in row and pd.notna(row[rel_col]) else np.nan
        d = abs(float(row[dlt_col])) if dlt_col in row and pd.notna(row[dlt_col]) else np.nan
        relerr[c].append(r)
        abserr[c].append(d)

fig, axs = plt.subplots(3, 1, figsize=(max(12, len(short_scenes)*0.7), 9), sharex=True)

max_abs_err = max(max(vals) for vals in abserr.values() if len(vals) > 0)

for ax, c in zip(axs, panel_order):
    xi = np.arange(len(short_scenes))

    # Left axis: relative error (grey bars)
    y_rel = np.array(relerr[c], dtype=float)
    ax.bar(xi, y_rel, width=0.6, color="lightgrey", edgecolor="black", alpha=0.85)
    ax.set_ylim(0, 5)
    ax.set_ylabel("Relative error", color="black")
    ax.tick_params(axis='y', colors="black")
    ax.set_title(CLASS_NAMES[c].capitalize(), pad=4)
    ax.grid(False)

    # Right axis: absolute pixel error (maroon points, no line)
    ax2 = ax.twinx()
    y_abs = np.array(abserr[c], dtype=float)
    ax2.scatter(xi, y_abs, color="maroon", s=20, zorder=3)
    ax2.set_ylim(0, max_abs_err * 1.05)
    ax2.set_ylabel("Absolute error (pixels)", color="maroon")
    ax2.tick_params(axis='y', colors="maroon")
    ax2.spines["right"].set_alpha(0.4)
    ax2.grid(False)

axs[-1].set_xticks(np.arange(len(short_scenes)))
axs[-1].set_xticklabels(short_scenes, rotation=0, ha='center')
axs[-1].set_xlabel("Ice shelf (Scene)")

err_path = OUT_DIR / "absolute_error_dualaxis_tripanel.png"
fig.tight_layout()
fig.savefig(err_path)
plt.close(fig)
print("Saved:", err_path)
