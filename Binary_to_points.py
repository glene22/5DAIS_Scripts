import os
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from shapely.geometry import Point
from joblib import Parallel, delayed

# --- CONFIGURATION ---
mask_folder = "/home/glene/luna/CPOM/glene/PostDoc/5DAIS/Methods/ML/Training/Thresholded/Training_Masks_copy"
samples_per_class = 2500
gee_csv_out = "/home/glene/luna/CPOM/glene/PostDoc/5DAIS/Methods/ML/Training/Training_Masks_copy_2500_test.csv"
n_jobs = 4  # safe parallelism for shared server

# --- FIND LAKE & SLUSH MASK PAIRS ---
lake_files = [f for f in os.listdir(mask_folder) if f.lower().startswith("modified_lakemask")]
slush_files = [f for f in os.listdir(mask_folder) if "slushmask" in f.lower()]

def extract_basename(fname):
    return fname.lower().replace("modified_lakemask_", "").replace("modified_slushmask_", "").replace("slushmask_", "").replace(".tif", "")

lake_dict = {extract_basename(f): f for f in lake_files}
slush_dict = {extract_basename(f): f for f in slush_files}
shared_keys = sorted(set(lake_dict.keys()) & set(slush_dict.keys()))

print(f"🔍 Found {len(shared_keys)} lake–slush pairs to process.\n")

# --- FUNCTION TO PROCESS A SINGLE PAIR ---
def process_pair(i, key):
    lake_path = os.path.join(mask_folder, lake_dict[key])
    slush_path = os.path.join(mask_folder, slush_dict[key])
    print(f"🔄 [{i+1}/{len(shared_keys)}] Processing: {key}")

    with rasterio.open(lake_path) as lake_src, rasterio.open(slush_path) as slush_src:
        lake = lake_src.read(1)
        slush = slush_src.read(1)

        if lake.shape != slush.shape:
            print(f"⚠️ Shape mismatch for {key}")
            return None

        transform = lake_src.transform
        crs = lake_src.crs

        class_mask = np.full_like(lake, 0)
        class_mask[(lake == 1) & (slush != 1)] = 1  # lake only
        class_mask[(slush == 1) & (lake != 1)] = 2  # slush only

        points = []
        for class_value in [0, 1, 2]:
            ys, xs = np.where(class_mask == class_value)
            if len(xs) == 0:
                continue
            idx = np.random.choice(len(xs), min(samples_per_class, len(xs)), replace=False)
            xs_sampled, ys_sampled = xs[idx], ys[idx]
            for x, y in zip(xs_sampled, ys_sampled):
                x_coord, y_coord = transform * (x, y)
                points.append({
                    "x": x_coord,
                    "y": y_coord,
                    "class": class_value,
                    "pair_id": key
                })

        if points:
            gdf = gpd.GeoDataFrame(points, geometry=[Point(p["x"], p["y"]) for p in points], crs=crs)
            return gdf.to_crs("EPSG:4326")
        else:
            return None

# --- PARALLEL EXECUTION ---
results = Parallel(n_jobs=n_jobs)(delayed(process_pair)(i, key) for i, key in enumerate(shared_keys))
gdf_all = [gdf for gdf in results if gdf is not None]

print("\n📦 Merging and exporting sampled points...")

# --- MERGE & EXPORT CSV ONLY ---
combined = gpd.GeoDataFrame(pd.concat(gdf_all, ignore_index=True), crs="EPSG:4326")
combined["longitude"] = combined.geometry.x
combined["latitude"] = combined.geometry.y

combined[["longitude", "latitude", "class", "pair_id"]].to_csv(gee_csv_out, index=False)

print(f"✅ Sampled points saved:\n📄 CSV: {gee_csv_out}")
