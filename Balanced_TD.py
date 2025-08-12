import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform

# --- CONFIGURATION ---
mask_folder = "/home/glene/luna/CPOM/glene/PostDoc/5DAIS/Methods/ML/Training/Thresholded/Training_Masks"
samples_per_class = {0: 5000, 1: 2500, 2: 2500}  # More background to reduce melt overprediction

gee_csv_out = "/home/glene/luna/CPOM/glene/PostDoc/5DAIS/Methods/ML/Training/Multiclass_thresholded_points_balanced.csv"

# --- FIND LAKE & SLUSH MASK PAIRS ---
lake_files = [f for f in os.listdir(mask_folder) if f.lower().startswith("modified_lakemask")]
slush_files = [f for f in os.listdir(mask_folder) if "slushmask" in f.lower()]

def extract_basename(fname):
    return fname.lower().replace("modified_lakemask_", "").replace("modified_slushmask_", "").replace("slushmask_", "").replace(".tif", "")

lake_dict = {extract_basename(f): f for f in lake_files}
slush_dict = {extract_basename(f): f for f in slush_files}
shared_keys = sorted(set(lake_dict.keys()) & set(slush_dict.keys()))

print(f"🔍 Found {len(shared_keys)} lake–slush pairs to process.\n")

# --- SAMPLE POINTS ---
all_points = []

for i, key in enumerate(shared_keys, 1):
    lake_path = os.path.join(mask_folder, lake_dict[key])
    slush_path = os.path.join(mask_folder, slush_dict[key])

    print(f"🔄 [{i}/{len(shared_keys)}] Processing: {key}")

    with rasterio.open(lake_path) as lake_src, rasterio.open(slush_path) as slush_src:
        lake = lake_src.read(1)
        slush = slush_src.read(1)

        assert lake.shape == slush.shape, f"Shape mismatch for {key}"
        transform_affine = lake_src.transform
        crs = lake_src.crs

        # Build multiclass mask
        melt_mask = np.zeros_like(lake, dtype=np.uint8)
        melt_mask[lake == 1] = 1
        melt_mask[slush == 1] = 2

        for class_value in [0, 1, 2]:
            ys, xs = np.where(melt_mask == class_value)
            count_available = len(xs)
            if count_available == 0:
                print(f"⚠️ No pixels found for class {class_value}")
                continue

            n_samples = min(samples_per_class[class_value], count_available)
            idx = np.random.choice(count_available, n_samples, replace=False)
            xs_sampled, ys_sampled = xs[idx], ys[idx]

            # Convert to projected coords
            proj_xs, proj_ys = rasterio.transform.xy(transform_affine, ys_sampled, xs_sampled)
            # Convert to lon/lat
            lons, lats = transform(crs, 'EPSG:4326', proj_xs, proj_ys)

            for lon, lat in zip(lons, lats):
                all_points.append({
                    "longitude": lon,
                    "latitude": lat,
                    "class": class_value,
                    "pair_id": key
                })
            print(f"✅ Sampled {n_samples} points from class {class_value}")

# --- EXPORT TO CSV ---
print("\n📦 Exporting sampled points...")
df = pd.DataFrame(all_points)
df.to_csv(gee_csv_out, index=False)
print(f"✅ CSV saved to: {gee_csv_out}")
