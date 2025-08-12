import os
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from shapely.geometry import Point

# --- CONFIGURATION ---
mask_folder = "/home/glene/luna/CPOM/glene/PostDoc/5DAIS/Methods/ML/Training/Thresholded/tm_crevasse"
samples_per_class = 2500

gee_csv_out = "/home/glene/luna/CPOM/glene/PostDoc/5DAIS/Methods/ML/Training/thresholded_tm_crevasse_2500_test.csv"
shp_out = "/home/glene/luna/CPOM/glene/PostDoc/5DAIS/Methods/ML/Training/thresholded_tm_crevasse_epsg4326.shp"

# --- FIND LAKE & SLUSH MASK PAIRS ---
lake_files = [f for f in os.listdir(mask_folder) if f.lower().startswith("modified_lakemask")]
slush_files = [f for f in os.listdir(mask_folder) if "slushmask" in f.lower()]

def extract_basename(fname):
    return fname.lower().replace("modified_lakemask_", "").replace("modified_slushmask_", "").replace("slushmask_", "").replace(".tif", "")

lake_dict = {extract_basename(f): f for f in lake_files}
slush_dict = {extract_basename(f): f for f in slush_files}
shared_keys = sorted(set(lake_dict.keys()) & set(slush_dict.keys()))

print(f"🔍 Found {len(shared_keys)} lake–slush pairs to process.\n")

# --- SAMPLE POINTS FROM MELT MASK ---
all_gdfs = []

for i, key in enumerate(shared_keys, 1):
    lake_path = os.path.join(mask_folder, lake_dict[key])
    slush_path = os.path.join(mask_folder, slush_dict[key])

    print(f"🔄 [{i}/{len(shared_keys)}] Processing: {key}")

    with rasterio.open(lake_path) as lake_src, rasterio.open(slush_path) as slush_src:
        lake = lake_src.read(1)
        slush = slush_src.read(1)

        assert lake.shape == slush.shape, f"Shape mismatch for {key}"
        melt_mask = np.where((lake == 1) | (slush == 1), 1, 0)

        transform = lake_src.transform
        crs = lake_src.crs

        points = []
        for class_value in [0, 1]:
            ys, xs = np.where(melt_mask == class_value)
            count_available = len(xs)
            if count_available == 0:
                print(f"⚠️ No pixels found for class {class_value}")
                continue
            idx = np.random.choice(count_available, min(samples_per_class, count_available), replace=False)
            xs_sampled, ys_sampled = xs[idx], ys[idx]
            for x, y in zip(xs_sampled, ys_sampled):
                x_coord, y_coord = transform * (x, y)
                points.append({
                    "x": x_coord,
                    "y": y_coord,
                    "class": class_value,
                    "pair_id": key
                })
            print(f"✅ Sampled {len(xs_sampled)} points from class {class_value}")

        gdf = gpd.GeoDataFrame(points, geometry=[Point(p["x"], p["y"]) for p in points], crs=crs)
        gdf_wgs84 = gdf.to_crs("EPSG:4326")
        all_gdfs.append(gdf_wgs84)

print("\n📦 Merging and exporting sampled points...")

# --- MERGE & EXPORT ---
combined = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True), crs="EPSG:4326")
combined["longitude"] = combined.geometry.x
combined["latitude"] = combined.geometry.y

combined[["longitude", "latitude", "class", "pair_id"]].to_csv(gee_csv_out, index=False)
combined.to_file(shp_out)

print(f"✅ Sampled points saved:\n📄 CSV: {gee_csv_out}\n🗂 SHP: {shp_out}")
