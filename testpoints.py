import os
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from shapely.geometry import Point

# --- CONFIGURATION ---
tif_folder = "/home/glene/luna/CPOM/glene/PostDoc/5DAIS/Methods/ML/Training/Lakes4Antarctica/train-test-data-Lancaster/test"
samples_per_class = 2500

gee_csv_out = "/home/glene/luna/CPOM/glene/PostDoc/5DAIS/Methods/ML/Training/test_points_2500.csv"
shp_out = "/home/glene/luna/CPOM/glene/PostDoc/5DAIS/Methods/ML/Training/test_points_epsg4326.shp"

# --- SAMPLE POINTS ---
all_gdfs = []

for fname in os.listdir(tif_folder):
    if not fname.endswith(".tif"):
        continue

    path = os.path.join(tif_folder, fname)
    with rasterio.open(path) as src:
        arr = src.read(1)
        transform = src.transform
        crs = src.crs

        print(f"📂 File: {fname}")
        print(f"📌 CRS: {crs}")
        print(f"📏 Shape: {arr.shape}")
        print(f"🗺 Transform: {transform}")
        print()

        points = []
        for class_value in [0, 1]:
            ys, xs = np.where(arr == class_value)
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
                    "source_file": fname
                })

        gdf = gpd.GeoDataFrame(points, geometry=[Point(p["x"], p["y"]) for p in points], crs=crs)
        gdf_wgs84 = gdf.to_crs("EPSG:4326")
        all_gdfs.append(gdf_wgs84)

# --- Merge all and export ---
combined = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True), crs="EPSG:4326")

# Save CSV
combined["longitude"] = combined.geometry.x
combined["latitude"] = combined.geometry.y
combined[["longitude", "latitude", "class", "source_file"]].to_csv(gee_csv_out, index=False)
print(f"✅ GEE CSV saved: {gee_csv_out}")

# Save shapefile
combined.to_file(shp_out)
print(f"✅ Shapefile saved: {shp_out}")
