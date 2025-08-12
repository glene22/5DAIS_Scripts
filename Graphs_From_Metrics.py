#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === EDIT THESE ===
CSV_PATH = "/home/glene/luna/CPOM/glene/PostDoc/5DAIS/Methods/ML/GEE/csv/metrics_fullscene.csv"
OUT_DIR = Path("/home/glene/luna/CPOM/glene/PostDoc/5DAIS/Methods/ML/comparison_figures")
CLASS_NAMES = ["background", "lake", "slush"]
# ==================

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load CSV
df = pd.read_csv(CSV_PATH, sep=None, engine="python")

# Convert numerics
for c in df.columns:
    if c not in ("Scene", "Pair"):
        try:
            df[c] = pd.to_numeric(df[c])
        except Exception:
            pass

# === Per-class metric bar plots in one figure ===
metrics_by_prefix = {
    "Precision": "Prec_class",
    "Recall": "Rec_class",
    "F1": "F1_class",
    "IoU": "IoU_class",
}

sns.set_palette("pastel")
scenes = df["Scene"].astype(str).tolist()
x = np.arange(len(scenes)) * 1.5  # increase spacing between groups
bar_width = 0.15  # narrower bars

fig, axes = plt.subplots(3, 1, figsize=(max(10, len(scenes)*0.8), 9), sharex=True)

for class_idx, ax in enumerate(axes):
    for i, (label, pref) in enumerate(metrics_by_prefix.items()):
        col = f"{pref}{class_idx}"
        if col in df.columns:
            y = df[col].astype(float).values
            ax.bar(x + i*bar_width - (1.5*bar_width), y, width=bar_width, label=label)
    ax.set_ylim(0.75, 1.0)
    ax.set_ylabel(CLASS_NAMES[class_idx])
    ax.grid(True, linestyle='--', alpha=0.3)
    if class_idx == 0:
        ax.set_title("Per-class Precision / Recall / F1 / IoU across ice shelves")
    if class_idx == 1:
        ax.legend(ncol=4, fontsize=9, loc='lower left')

axes[-1].set_xticks(x)
axes[-1].set_xticklabels(scenes, rotation=60, ha='right')
axes[-1].set_xlabel("Ice shelf (Scene)")

plt.tight_layout()
plt.savefig(OUT_DIR / "per_class_metrics_pastel_bars.png", dpi=200)
plt.close()

print(f"Saved figure to: {OUT_DIR.resolve()}")