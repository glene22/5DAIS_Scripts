import os
import glob
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from math import ceil

# --- EDIT THESE ---
GT_DIR = "/home/glene/luna/CPOM/glene/PostDoc/5DAIS/Methods/ML/GEE/GT_masks"
CLF_DIR = "/home/glene/luna/CPOM/glene/PostDoc/5DAIS/Methods/ML/GEE/Classified_masks_cld"
OUT_CSV = "/home/glene/luna/CPOM/glene/PostDoc/5DAIS/Methods/ML/GEE/csv/mask_comparison_metrics_multiclass_sampled_stratified.csv"
FIG_DIR = "/home/glene/luna/CPOM/glene/PostDoc/5DAIS/Methods/ML/GEE/comparison_figures"
CLASSES = [0, 1, 2]                     # 0=background, 1=slush, 2=lake
CLASS_LABELS = {0:"Background", 1:"Slush", 2:"Lake"}
SAMPLE_SIZE = 60000                      # total sample size (even per class)
MAX_PLOT_SIZE = 2000
DPI = 200
# ------------------

os.makedirs(FIG_DIR, exist_ok=True)

def find_classified_partner(gt_path):
    base = os.path.basename(gt_path)
    suffix = base.replace("GT_", "").replace(".tif", "")
    exact = os.path.join(CLF_DIR, f"Classified_cld_{suffix}.tif")
    if os.path.exists(exact):
        return exact
    candidates = sorted(glob.glob(os.path.join(CLF_DIR, f"Classified_cld_{suffix}*.tif")))
    return candidates[0] if candidates else None

def read_and_align(gt_path, clf_path):
    with rasterio.open(gt_path) as gt_ds:
        gt = gt_ds.read(1)
        gt_nodata = gt_ds.nodata
        gt_transform = gt_ds.transform
        gt_crs = gt_ds.crs
        gt_shape = gt_ds.shape

    with rasterio.open(clf_path) as cl_ds:
        cl = cl_ds.read(1)
        cl_nodata = cl_ds.nodata
        cl_on_gt = np.zeros(gt_shape, dtype=cl.dtype)
        cl_on_gt[:] = cl_nodata if cl_nodata is not None else 0

        reproject(
            source=cl,
            destination=cl_on_gt,
            src_transform=cl_ds.transform,
            src_crs=cl_ds.crs,
            dst_transform=gt_transform,
            dst_crs=gt_crs,
            resampling=Resampling.nearest,
            dst_nodata=cl_nodata
        )

    valid = np.ones(gt_shape, dtype=bool)
    if gt_nodata is not None:
        valid &= (gt != gt_nodata)
    if cl_nodata is not None:
        valid &= (cl_on_gt != cl_nodata)
    valid &= np.isfinite(gt) & np.isfinite(cl_on_gt)

    return gt.astype(np.int16), cl_on_gt.astype(np.int16), valid

def sample_pixels_stratified(gt, pred, valid, n_samples_total, classes):
    """Evenly sample pixels from each class in GT."""
    per_class = n_samples_total // len(classes)
    gt_samples, pred_samples = [], []

    for cls in classes:
        y, x = np.where(valid & (gt == cls))
        total = len(y)
        if total == 0:
            continue  # class not present
        if total > per_class:
            idx = np.random.randint(0, total, size=per_class)
        else:
            idx = np.arange(total)  # take all if fewer
        gt_samples.append(gt[y[idx], x[idx]])
        pred_samples.append(pred[y[idx], x[idx]])

    if gt_samples:
        return np.concatenate(gt_samples), np.concatenate(pred_samples)
    else:
        return np.array([]), np.array([])

def compute_metrics(gt, pred, classes):
    n = len(classes)
    cm = np.zeros((n, n), dtype=np.int64)
    for i, c_t in enumerate(classes):
        for j, c_p in enumerate(classes):
            cm[i, j] = np.sum((gt == c_t) & (pred == c_p))

    precisions, recalls, f1s, ious = [], [], [], []
    for i in range(n):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP
        prec = TP / (TP + FP) if (TP + FP) else 0
        rec = TP / (TP + FN) if (TP + FN) else 0
        f1  = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        iou = TP / (TP + FP + FN) if (TP + FP + FN) else 0
        precisions.append(prec); recalls.append(rec); f1s.append(f1); ious.append(iou)

    total = cm.sum()
    acc = np.trace(cm) / total if total else np.nan
    mean_iou = float(np.mean(ious))
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / (total**2) if total else 0
    kappa = (acc - pe) / (1 - pe) if (1 - pe) else np.nan

    metrics = {"Accuracy": acc, "Mean_IoU": mean_iou, "Kappa": kappa}
    for k, cls in enumerate(classes):
        metrics[f"Prec_class{cls}"] = precisions[k]
        metrics[f"Rec_class{cls}"]  = recalls[k]
        metrics[f"F1_class{cls}"]   = f1s[k]
    return cm, metrics

def downsample_for_plot(a, max_side=2000):
    h, w = a.shape
    scale = max(h, w)
    if scale <= max_side:
        return a
    stride = ceil(scale / max_side)
    return a[::stride, ::stride]

def make_colormaps():
    class_colors = ListedColormap([
        "#c0c0c0", "#1b9e77", "#7570b3"
    ])
    class_norm = BoundaryNorm(np.arange(-0.5, len(CLASSES)+0.5, 1), class_colors.N)

    err_colors = ListedColormap([
        "#f0f0f0", "#e41a1c", "#377eb8",
        "#a65628", "#984ea3", "#4daf4a", "#ff7f00"
    ])
    err_norm = BoundaryNorm(np.arange(-0.5, 6.5+1, 1), err_colors.N)
    err_labels = [
        "Correct", "BG→Slush", "BG→Lake",
        "Slush→BG", "Slush→Lake", "Lake→BG", "Lake→Slush"
    ]
    return (class_colors, class_norm), (err_colors, err_norm, err_labels)

def build_error_map(gt, pred):
    err = np.zeros_like(gt, dtype=np.uint8)
    err[(gt==0) & (pred==1)] = 1
    err[(gt==0) & (pred==2)] = 2
    err[(gt==1) & (pred==0)] = 3
    err[(gt==1) & (pred==2)] = 4
    err[(gt==2) & (pred==0)] = 5
    err[(gt==2) & (pred==1)] = 6
    return err

def plot_triptych(name, gt, pred, outdir, metrics):
    (cls_cmap, cls_norm), (err_cmap, err_norm, err_labels) = make_colormaps()
    err = build_error_map(gt, pred)
    gt_d   = downsample_for_plot(gt, MAX_PLOT_SIZE)
    pred_d = downsample_for_plot(pred, MAX_PLOT_SIZE)
    err_d  = downsample_for_plot(err, MAX_PLOT_SIZE)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=DPI, constrained_layout=True)
    axes[0].imshow(gt_d, cmap=cls_cmap, norm=cls_norm, interpolation='nearest')
    axes[0].set_title("Ground Truth"); axes[0].axis('off')
    axes[1].imshow(pred_d, cmap=cls_cmap, norm=cls_norm, interpolation='nearest')
    axes[1].set_title("Prediction"); axes[1].axis('off')
    axes[2].imshow(err_d, cmap=err_cmap, norm=err_norm, interpolation='nearest')
    axes[2].set_title("Error Map"); axes[2].axis('off')

    class_handles = [plt.Line2D([0],[0], marker='s', linestyle='',
                                 markerfacecolor=cls_cmap(k), markeredgecolor='k')
                     for k in range(len(CLASSES))]
    class_labels = [CLASS_LABELS.get(k, f"Class {k}") for k in CLASSES]
    axes[0].legend(class_handles, class_labels, loc="lower left", fontsize=8, frameon=True)

    err_handles = [plt.Line2D([0],[0], marker='s', linestyle='',
                               markerfacecolor=err_cmap(k), markeredgecolor='k')
                   for k in range(7)]
    axes[2].legend(err_handles, err_labels, loc="lower left", fontsize=8, frameon=True)

    fig.suptitle(f"{name}  |  Acc={metrics['Accuracy']:.3f}  mIoU={metrics['Mean_IoU']:.3f}  Kappa={metrics['Kappa']:.3f}",
                 fontsize=12)

    png_path = os.path.join(outdir, f"{name}_comparison.png")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return png_path

def main():
    np.random.seed(42)
    rows = []
    files = sorted(glob.glob(os.path.join(GT_DIR, "GT_*.tif")))
    if not files:
        print(f"No GT files in {GT_DIR}")
        return

    for gt_path in files:
        name = os.path.basename(gt_path).replace(".tif", "")
        clf_path = find_classified_partner(gt_path)
        if not clf_path:
            print(f"[WARN] No Classified match for {name}")
            continue

        gt_arr, cl_arr, valid = read_and_align(gt_path, clf_path)
        vcount = int(np.count_nonzero(valid))
        print(f"{name}: {vcount} valid pixels")
        if vcount == 0:
            continue

        # stratified sample for metrics
        gt_s, cl_s = sample_pixels_stratified(gt_arr, cl_arr, valid, SAMPLE_SIZE, CLASSES)
        if gt_s.size == 0:
            print(f"[WARN] No samples for {name}")
            continue

        cm, metrics = compute_metrics(gt_s, cl_s, CLASSES)
        png_path = plot_triptych(name, gt_arr, cl_arr, FIG_DIR, metrics)
        print(f"[OK] {name}: Acc={metrics['Accuracy']:.3f}, mIoU={metrics['Mean_IoU']:.3f}  →  {png_path}")

        row = {"Pair": f"{os.path.basename(gt_path)} vs {os.path.basename(clf_path)}"}
        row.update(metrics)
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(OUT_CSV, index=False)
        print(f"\nSaved metrics table: {OUT_CSV}")
        print(f"Figures in: {FIG_DIR}")

if __name__ == "__main__":
    main()
