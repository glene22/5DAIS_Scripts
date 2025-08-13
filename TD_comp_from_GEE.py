import os
import glob
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from math import ceil
import itertools

# --- EDIT THESE ---
GT_DIR = "/home/glene/luna/CPOM/glene/PostDoc/5DAIS/Methods/ML/GEE/Masks/GT_masks_NEW"
CLF_DIR = "/home/glene/luna/CPOM/glene/PostDoc/5DAIS/Methods/ML/GEE/Masks/Classified_NEW"
OUT_DIR = "/home/glene/luna/CPOM/glene/PostDoc/5DAIS/Methods/ML/GEE/csv"  # new: directory for CSVs
FIG_DIR = "/home/glene/luna/CPOM/glene/PostDoc/5DAIS/Methods/ML/GEE/comparison_figures"
CLASSES = [0, 1, 2]                     # 0=background, 1=slush, 2=lake (keep origin)
CLASS_LABELS = {0:"Background", 1:"Slush", 2:"Lake"}
SAMPLES_PER_CLASS = 50000               # cap per class for metrics sampling
MAX_PLOT_SIZE = 2000
DPI = 200
PLOT_CRS = "EPSG:3031"                  # Antarctic Polar Stereographic for figures

# Zoom configuration
SAVE_ZOOM = True                         # save an additional zoomed triptych
ZOOM_MODE = "meltwater"                  # 'meltwater' | 'error' | 'pred_meltwater' | 'nonbg'
ZOOM_FRAC = 0.35                         # window height/width as fraction of min(H,W)
ZOOM_PAD = 100                           # extra pixels around hotspot window
# ------------------

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# derived CSV paths (requested)
OUT_CSV_FULL = os.path.join(OUT_DIR, "metrics_fullscene.csv")
OUT_CSV_MW   = os.path.join(OUT_DIR, "metrics_meltwateronly.csv")


def find_classified_partner(gt_path):
    base = os.path.basename(gt_path)
    suffix = base.replace("GT_", "").replace(".tif", "")
    exact = os.path.join(CLF_DIR, f"Classified_cld_{suffix}.tif")
    if os.path.exists(exact):
        return exact
    candidates = sorted(glob.glob(os.path.join(CLF_DIR, f"Classified_cld_{suffix}*.tif")))
    return candidates[0] if candidates else None


def read_and_align(gt_path, clf_path):
    """Read GT and reproject Classified to GT grid; return arrays + GT CRS/transform/shape."""
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

    return gt.astype(np.int16), cl_on_gt.astype(np.int16), valid, gt_crs, gt_transform, gt_shape


def read_and_align_mosaic(gt_path, clf_paths):
    """Reproject multiple classified tiles to GT grid and mosaic; return arrays + GT CRS/transform/shape."""
    with rasterio.open(gt_path) as gt_ds:
        gt = gt_ds.read(1)
        gt_nodata = gt_ds.nodata
        gt_transform = gt_ds.transform
        gt_crs = gt_ds.crs
        gt_shape = gt_ds.shape

    temp_nodata = np.int16(-32768)
    cl_on_gt = np.full(gt_shape, temp_nodata, dtype=np.int16)

    for p in clf_paths:
        with rasterio.open(p) as cl_ds:
            tmp = np.full(gt_shape, temp_nodata, dtype=np.int16)
            reproject(
                source=cl_ds.read(1),
                destination=tmp,
                src_transform=cl_ds.transform,
                src_crs=cl_ds.crs,
                dst_transform=gt_transform,
                dst_crs=gt_crs,
                resampling=Resampling.nearest,
                dst_nodata=temp_nodata
            )
        fill = (cl_on_gt == temp_nodata) & (tmp != temp_nodata)
        cl_on_gt[fill] = tmp[fill]

    valid = np.ones(gt_shape, dtype=bool)
    if gt_nodata is not None:
        valid &= (gt != gt_nodata)
    valid &= (cl_on_gt != temp_nodata)
    valid &= np.isfinite(gt) & np.isfinite(cl_on_gt)

    return gt.astype(np.int16), cl_on_gt.astype(np.int16), valid, gt_crs, gt_transform, gt_shape


def sample_pixels_stratified(gt, pred, valid, samples_per_class, classes):
    """
    Evenly sample exactly `samples_per_class` pixels per class from valid GT.
    Uses sampling WITH replacement when a class has fewer pixels than requested.
    If a class is absent in GT for this scene, it contributes zero samples.
    """
    # independent RNG (doesn't rely on global np.random seed)
    rng = np.random.default_rng(42)

    gt_samples, pred_samples = [], []
    for cls in classes:
        y, x = np.where(valid & (gt == cls))
        total = len(y)
        if total == 0:
            # Class absent in GT: skip; will become a zero row/col in CM.
            continue

        # choose indices WITH replacement if needed
        idx = rng.choice(total, size=samples_per_class, replace=(total < samples_per_class))
        gt_samples.append(gt[y[idx], x[idx]])
        pred_samples.append(pred[y[idx], x[idx]])

    if gt_samples:
        return np.concatenate(gt_samples), np.concatenate(pred_samples)
    else:
        # no valid samples at all
        return np.array([], dtype=gt.dtype), np.array([], dtype=pred.dtype)



def compute_cm(gt, pred, classes, valid_mask=None):
    """Confusion matrix (raw pixel counts), optional valid mask."""
    if valid_mask is None:
        valid_mask = np.ones_like(gt, dtype=bool)
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for i, c_t in enumerate(classes):
        tmask = (gt == c_t) & valid_mask
        for j, c_p in enumerate(classes):
            cm[i, j] = np.count_nonzero(tmask & (pred == c_p))
    return cm


def metrics_from_sample(gt_s, pred_s, classes):
    """Metrics from the stratified sample (precision/recall/F1/IoU, etc.)."""
    n = len(classes)
    cm = np.zeros((n, n), dtype=np.int64)
    for i, c_t in enumerate(classes):
        for j, c_p in enumerate(classes):
            cm[i, j] = np.sum((gt_s == c_t) & (pred_s == c_p))

    precisions, recalls, f1s, ious = [], [], [], []
    for i in range(n):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP
        prec = TP / (TP + FP) if (TP + FP) else 0
        rec  = TP / (TP + FN) if (TP + FN) else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        iou  = TP / (TP + FP + FN) if (TP + FP + FN) else 0
        precisions.append(prec); recalls.append(rec); f1s.append(f1); ious.append(iou)

    total = cm.sum()
    acc = np.trace(cm) / total if total else np.nan
    mean_iou = float(np.mean(ious)) if ious else np.nan
    meltwater_iou = float(np.mean([ious[i] for i, c in enumerate(classes) if c in (1, 2)])) if ious else np.nan
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / (total**2) if total else 0
    kappa = (acc - pe) / (1 - pe) if (1 - pe) else np.nan

    metrics = {
        "Accuracy": acc,
        "Mean_IoU": mean_iou,
        "Meltwater_mIoU": meltwater_iou,
        "Kappa": kappa
    }
    for k, cls in enumerate(classes):
        metrics[f"Prec_class{cls}"] = precisions[k]
        metrics[f"Rec_class{cls}"]  = recalls[k]
        metrics[f"F1_class{cls}"]   = f1s[k]
        metrics[f"IoU_class{cls}"]  = ious[k]
    return cm, metrics


def downsample_for_plot(a, max_side=2000):
    h, w = a.shape
    scale = max(h, w)
    if scale <= max_side:
        return a
    stride = ceil(scale / max_side)
    return a[::stride, ::stride]


def make_colormaps():
    class_colors = ListedColormap(["#c0c0c0", "#1b9e77", "#7570b3"])  # BG, Slush, Lake
    class_norm = BoundaryNorm(np.arange(-0.5, len(CLASSES)+0.5, 1), class_colors.N)
    err_colors = ListedColormap(["#f0f0f0", "#e41a1c", "#377eb8", "#a65628", "#984ea3", "#4daf4a", "#ff7f00"])
    err_norm = BoundaryNorm(np.arange(-0.5, 6.5, 1), err_colors.N)  # categories 0..6
    err_labels = ["Correct", "BG→Slush", "BG→Lake", "Slush→BG", "Slush→Lake", "Lake→BG", "Lake→Slush"]
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


def reproject_arrays_to_crs(arr_list, src_crs, src_transform, src_shape, dst_crs):
    """
    Reproject one or more single-band arrays (category labels) from a common src grid
    to a common dst grid defined by calculate_default_transform(src_crs -> dst_crs) using
    GT's bounds/shape. Returns (dst_arrays, dst_transform, dst_width, dst_height).
    """
    height, width = src_shape
    left = src_transform.c
    top = src_transform.f
    right = left + src_transform.a * width
    bottom = top + src_transform.e * height

    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, width, height, left, bottom, right, top
    )

    dst_arrays = []
    for arr in arr_list:
        dst = np.full((dst_height, dst_width), 0, dtype=arr.dtype)
        reproject(
            source=arr,
            destination=dst,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest
        )
        dst_arrays.append(dst)
    return dst_arrays, dst_transform, dst_width, dst_height


# --------- NEW: Hotspot finder for zoom ---------
def find_hotspot_bbox(mask: np.ndarray, win_h: int, win_w: int, pad: int = 0):
    """
    Return (y0,y1,x0,x1) for the window with the maximum sum of True pixels.
    Uses an integral image so it's fast even for big rasters.
    """
    h, w = mask.shape
    win_h = min(win_h, h)
    win_w = min(win_w, w)
    # integral image
    S = np.zeros((h+1, w+1), dtype=np.int64)
    S[1:,1:] = np.cumsum(np.cumsum(mask.astype(np.int32), axis=0), axis=1)
    # possible top-left coords
    if (h - win_h + 1) <= 0 or (w - win_w + 1) <= 0:
        # window is whole image
        y0, x0 = 0, 0
    else:
        y_idx = np.arange(0, h - win_h + 1)
        x_idx = np.arange(0, w - win_w + 1)
        YY, XX = np.meshgrid(y_idx, x_idx, indexing='ij')
        sums = (S[YY + win_h, XX + win_w] - S[YY, XX + win_w]
                - S[YY + win_h, XX] + S[YY, XX])
        iy, ix = np.unravel_index(np.argmax(sums), sums.shape)
        y0, x0 = y_idx[iy], x_idx[ix]
    y1, x1 = y0 + win_h, x0 + win_w
    # pad and clip
    y0 = max(0, y0 - pad); x0 = max(0, x0 - pad)
    y1 = min(h, y1 + pad); x1 = min(w, x1 + pad)
    return y0, y1, x0, x1


def plot_triptych(scene_tag, gt, pred, outdir, metrics, src_crs, src_transform, src_shape,
                  save_zoom=SAVE_ZOOM, zoom_mode=ZOOM_MODE, zoom_frac=ZOOM_FRAC, zoom_pad=ZOOM_PAD):
    """Reproject GT and prediction to EPSG:3031 for plotting (figures only), then render triptych (+ optional crop)."""
    err_src = build_error_map(gt, pred)

    (gt_ps, pred_ps, err_ps), _, _, _ = reproject_arrays_to_crs(
        [gt, pred, err_src], src_crs, src_transform, src_shape, PLOT_CRS
    )

    (cls_cmap, cls_norm), (err_cmap, err_norm, err_labels) = make_colormaps()
    gt_d   = downsample_for_plot(gt_ps, MAX_PLOT_SIZE)
    pred_d = downsample_for_plot(pred_ps, MAX_PLOT_SIZE)
    err_d  = downsample_for_plot(err_ps, MAX_PLOT_SIZE)

    # ----- full scene -----
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

    fig.suptitle(
        f"{scene_tag}  |  Accuracy={metrics['Accuracy']:.3f}  "
        f"mIoU={metrics['Mean_IoU']:.3f}  Kappa={metrics['Kappa']:.3f}  "
        f"Melt mIoU={metrics['Meltwater_mIoU']:.3f}",
        fontsize=12
    )

    png_path = os.path.join(outdir, f"triptych_{scene_tag}.png")
    # fig.savefig(png_path, bbox_inches="tight")
    # plt.close(fig)

    # ----- cropped scene -----
    zoom_png = None
    if save_zoom:
        if zoom_mode == "meltwater":
            focus = (gt_ps == 1) | (gt_ps == 2)
        elif zoom_mode == "pred_meltwater":
            focus = (pred_ps == 1) | (pred_ps == 2)
        elif zoom_mode == "error":
            focus = (err_ps > 0)
        elif zoom_mode == "nonbg":
            focus = (gt_ps != 0)
        else:
            focus = (gt_ps != 0)

        if np.any(focus):
            H, W = focus.shape
            win = int(max(1, zoom_frac * min(H, W)))
            y0, y1, x0, x1 = find_hotspot_bbox(focus, win, win, pad=zoom_pad)

            gt_c   = gt_ps[y0:y1, x0:x1]
            pred_c = pred_ps[y0:y1, x0:x1]
            err_c  = err_ps[y0:y1, x0:x1]

            gt_cd   = downsample_for_plot(gt_c, MAX_PLOT_SIZE)
            pred_cd = downsample_for_plot(pred_c, MAX_PLOT_SIZE)
            err_cd  = downsample_for_plot(err_c, MAX_PLOT_SIZE)

            fig2, ax2 = plt.subplots(1, 3, figsize=(15, 5), dpi=DPI, constrained_layout=True)
            ax2[0].imshow(gt_cd, cmap=cls_cmap, norm=cls_norm, interpolation='nearest')
            ax2[0].set_title("Ground Truth"); ax2[0].axis('off')
            ax2[1].imshow(pred_cd, cmap=cls_cmap, norm=cls_norm, interpolation='nearest')
            ax2[1].set_title("Prediction"); ax2[1].axis('off')
            ax2[2].imshow(err_cd, cmap=err_cmap, norm=err_norm, interpolation='nearest')
            ax2[2].set_title("Error Map"); ax2[2].axis('off')

            ax2[0].legend(class_handles, class_labels, loc="lower left", fontsize=8, frameon=True)
            ax2[2].legend(err_handles, err_labels, loc="lower left", fontsize=8, frameon=True)

            fig2.suptitle(
                f"{scene_tag}  |  Accuracy={metrics['Accuracy']:.3f}  "
                f"mIoU={metrics['Mean_IoU']:.3f}  Kappa={metrics['Kappa']:.3f}  "
                f"Melt mIoU={metrics['Meltwater_mIoU']:.3f}",
                fontsize=12
            )

            zoom_png = os.path.join(outdir, f"triptych_{scene_tag}_crop.png")
            fig2.savefig(zoom_png, bbox_inches="tight")
            plt.close(fig2)
        else:
            print(f"[INFO] {scene_tag}: no focus pixels for crop ({zoom_mode}).")

    return png_path, zoom_png



def plot_confusion_matrix(cm, class_names, title, outpath):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=DPI)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(class_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           title=title, ylabel='True label', xlabel='Predicted label')
    thresh = cm.max() / 2. if cm.max() > 0 else 0.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, f"{cm[i, j]:,d}", ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    # fig.savefig(outpath, bbox_inches="tight")
    # plt.close(fig)


def main():
    np.random.seed(42)
    rows_full = []
    rows_mw = []

    files = sorted(glob.glob(os.path.join(GT_DIR, "GT_*.tif")))
    if not files:
        print(f"No GT files in {GT_DIR}")
        return

    # Determine class IDs by name for robust naming (handles mapping without changing indices)
    lake_id = next((k for k, v in CLASS_LABELS.items() if v.lower().startswith('lake')), 2)
    slush_id = next((k for k, v in CLASS_LABELS.items() if v.lower().startswith('slush')), 1)

    for gt_path in files:
        name = os.path.basename(gt_path).replace(".tif", "")  # e.g., GT_Abbott_...
        scene_tag = name.replace("GT_", "")

        if "Abbott" in name:
            abbott_tiles = sorted(glob.glob(os.path.join(CLF_DIR, "Classified_cld_Abbott_1_1-*.tif")))
            if not abbott_tiles:
                fallback = find_classified_partner(gt_path)
                abbott_tiles = [fallback] if fallback else []
            if not abbott_tiles:
                print(f"[WARN] No Classified tiles for {name}")
                continue
            print(f"{name}: mosaicking {len(abbott_tiles)} classified tiles")
            gt_arr, cl_arr, valid, gt_crs, gt_transform, gt_shape = read_and_align_mosaic(gt_path, abbott_tiles)
            clf_desc = ", ".join(os.path.basename(p) for p in abbott_tiles)
        else:
            clf_path = find_classified_partner(gt_path)
            if not clf_path:
                print(f"[WARN] No Classified match for {name}")
                continue
            gt_arr, cl_arr, valid, gt_crs, gt_transform, gt_shape = read_and_align(gt_path, clf_path)
            clf_desc = os.path.basename(clf_path)

        vcount = int(np.count_nonzero(valid))
        if vcount == 0:
            print(f"[WARN] {name}: 0 valid pixels")
            continue

        # --- FULL-SCENE CONFUSION (pixel counts) ---
        full_cm = compute_cm(gt_arr, cl_arr, CLASSES, valid_mask=valid)
        gt_counts = full_cm.sum(axis=1)
        pred_counts = full_cm.sum(axis=0)
        total_valid = int(gt_counts.sum())
        gt_perc_full = gt_counts / total_valid
        pred_perc_full = pred_counts / total_valid
        delta_counts = pred_counts - gt_counts
        rel_err = np.where(gt_counts > 0, delta_counts / gt_counts, np.nan)

        # --- STRATIFIED SAMPLE METRICS (full-scene) ---
        gt_s, cl_s = sample_pixels_stratified(gt_arr, cl_arr, valid, SAMPLES_PER_CLASS, CLASSES)
        if gt_s.size == 0:
            print(f"[WARN] No samples for {name}")
            continue
        sample_cm, metrics_full = metrics_from_sample(gt_s, cl_s, CLASSES)

        # Stratified confusion matrix (sampled)
        cm_full_strat_path = os.path.join(FIG_DIR, f"confusion_fullscene_stratified_{scene_tag}.png")
        plot_confusion_matrix(sample_cm, [CLASS_LABELS[c] for c in CLASSES],
                              title=f"{scene_tag} | Confusion (stratified sample, 3 classes)",
                              outpath=cm_full_strat_path)

        # Figures: triptych (EPSG:3031) and full-scene confusion
        triptych_path, zoom_path = plot_triptych(scene_tag, gt_arr, cl_arr, FIG_DIR, metrics_full,
                                                 gt_crs, gt_transform, gt_shape,
                                                 save_zoom=SAVE_ZOOM, zoom_mode=ZOOM_MODE,
                                                 zoom_frac=ZOOM_FRAC, zoom_pad=ZOOM_PAD)
        cm_full_path = os.path.join(FIG_DIR, f"confusion_fullscene_{scene_tag}.png")
        plot_confusion_matrix(full_cm, [CLASS_LABELS[c] for c in CLASSES],
                              title=f"{scene_tag} | Confusion (pixel counts, 3 classes)",
                              outpath=cm_full_path)

        ztxt = f", {os.path.basename(zoom_path)}" if zoom_path else ""
        print(
            f"[OK] {scene_tag}: Acc={metrics_full['Accuracy']:.3f}, mIoU={metrics_full['Mean_IoU']:.3f}, "
            f"Melt mIoU={metrics_full['Meltwater_mIoU']:.3f} → {triptych_path}{ztxt}, {cm_full_path}"
        )

        # Row for FULL CSV
        row_full = {
            "Scene": scene_tag,
            "Pair": f"{os.path.basename(gt_path)} vs {clf_desc}",
            "Total_valid_px": total_valid,
        }
        row_full.update(metrics_full)
        for i, c in enumerate(CLASSES):
            row_full[f"GT_pct_class{c}"] = float(gt_perc_full[i])
            row_full[f"Pred_pct_class{c}"] = float(pred_perc_full[i])
        for i, c in enumerate(CLASSES):
            row_full[f"GT_px_class{c}"] = int(gt_counts[i])
            row_full[f"Pred_px_class{c}"] = int(pred_counts[i])
            row_full[f"Delta_px_class{c}"] = int(delta_counts[i])
            row_full[f"RelErr_class{c}"] = float(rel_err[i]) if np.isfinite(rel_err[i]) else np.nan
        for ti, tc in enumerate(CLASSES):
            for pj, pc in enumerate(CLASSES):
                row_full[f"CM_strat_t{tc}_p{pc}"] = int(sample_cm[ti, pj])
        rows_full.append(row_full)

        # ===================== MELTWATER-ONLY =====================
        MW_CLASSES = [lake_id, slush_id]  # order by label names for clarity
        mw_valid = valid & np.isin(gt_arr, MW_CLASSES) & np.isin(cl_arr, MW_CLASSES)

        # 2x2 confusion (raw pixel counts) over lakes/slush only
        cm_mw = compute_cm(gt_arr, cl_arr, MW_CLASSES, valid_mask=mw_valid)
        # Stratified confusion over lakes/slush (sampled)
        cm_mw_strat_path = os.path.join(FIG_DIR, f"confusion_meltwateronly_stratified_{scene_tag}.png")
        cm_mw_path = os.path.join(FIG_DIR, f"confusion_meltwateronly_{scene_tag}.png")
        plot_confusion_matrix(cm_mw, [CLASS_LABELS[c] for c in MW_CLASSES],
                              title=f"{scene_tag} | Confusion (pixel counts, lakes/slush)",
                              outpath=cm_mw_path)
        # Stratified sampling metrics for meltwater-only
        gt_s_mw, cl_s_mw = sample_pixels_stratified(gt_arr, cl_arr, mw_valid, SAMPLES_PER_CLASS, MW_CLASSES)
        if gt_s_mw.size == 0:
            print(f"[WARN] {scene_tag}: no meltwater-only samples; skipping MW metrics")
            continue
        cm_mw_strat, metrics_mw = metrics_from_sample(gt_s_mw, cl_s_mw, MW_CLASSES)
        # Save stratified (sample-based) confusion matrix for meltwater-only
        plot_confusion_matrix(cm_mw_strat, [CLASS_LABELS[c] for c in MW_CLASSES],
                              title=f"{scene_tag} | Confusion (stratified sample, lakes/slush)",
                              outpath=cm_mw_strat_path)

        # Reported metrics: F1_lakes, F1_slush, macro, weighted (by GT pixel counts)
        F1_lakes = metrics_mw.get(f"F1_class{lake_id}", np.nan)
        F1_slush = metrics_mw.get(f"F1_class{slush_id}", np.nan)
        F1_macro = np.nanmean([F1_lakes, F1_slush])
        # weights from raw GT pixel counts within MW-valid mask
        gt_counts_mw = cm_mw.sum(axis=1).astype(float)
        weight_lake = gt_counts_mw[MW_CLASSES.index(lake_id)] / gt_counts_mw.sum() if gt_counts_mw.sum() else np.nan
        weight_slush = gt_counts_mw[MW_CLASSES.index(slush_id)] / gt_counts_mw.sum() if gt_counts_mw.sum() else np.nan
        F1_weighted = (
            F1_lakes * weight_lake + F1_slush * weight_slush
        ) if np.all(np.isfinite([F1_lakes, F1_slush, weight_lake, weight_slush])) else np.nan

        row_mw = {
            "Scene": scene_tag,
            "Pair": f"{os.path.basename(gt_path)} vs {clf_desc}",
            "Total_valid_px_mw": int(gt_counts_mw.sum()),
            # carry core metrics but computed over MW_CLASSES only
            "Accuracy": metrics_mw["Accuracy"],
            "Mean_IoU": metrics_mw["Mean_IoU"],
            "Kappa": metrics_mw["Kappa"],
            # requested
            "F1_lakes": F1_lakes,
            "F1_slush": F1_slush,
            "F1_meltwater_macro": F1_macro,
            "F1_meltwater_weighted": F1_weighted,
        }
        # also include per-class Prec/Rec/IoU for completeness
        for key in ["Prec", "Rec", "F1", "IoU"]:
            for cls in MW_CLASSES:
                row_mw[f"{key}_class{cls}"] = metrics_mw.get(f"{key}_class{cls}", np.nan)

        rows_mw.append(row_mw)

    # ===================== SAVE TABLES =====================
    if rows_full:
        df_full = pd.DataFrame(rows_full)
        df_full.to_csv(OUT_CSV_FULL, index=False)
        print(f"\nSaved full-scene metrics: {OUT_CSV_FULL}")
    else:
        print("No full-scene rows to save.")

    if rows_mw:
        df_mw = pd.DataFrame(rows_mw)
        df_mw.to_csv(OUT_CSV_MW, index=False)
        print(f"Saved meltwater-only metrics: {OUT_CSV_MW}")
    else:
        print("No meltwater-only rows to save.")

    print(f"Figures in: {FIG_DIR}")


if __name__ == "__main__":
    main()
