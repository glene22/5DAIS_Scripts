import os
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from shapely.geometry import Point, Polygon
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
mask_folder = "/home/glene/luna/CPOM/glene/PostDoc/5DAIS/Methods/ML/Training/Thresholded/tm_crevasse"
samples_per_class = 2500
gee_csv_out = "/home/glene/luna/CPOM/glene/PostDoc/5DAIS/Methods/ML/Training/tm_crevasse_bounding.csv"
visual_out = "/home/glene/luna/CPOM/glene/PostDoc/5DAIS/Methods/ML/Training/tm_crevasse_bounding_preview.png"
n_jobs = 3

# --- DEFINE BOUNDING BOXES (in EPSG:4326) ---
print("📦 Defining bounding boxes...")
bounds_dict = {
    "hull_2018": Polygon([
        (-137.52170886192363, -75.07681503907241),
        (-136.894114867783, -75.07681503907241),
        (-136.894114867783, -74.9454004931143),
        (-137.52170886192363, -74.9454004931143),
        (-137.52170886192363, -75.07681503907241)
    ]),
    "nickerson_2018": Polygon([
        (-142.25471206460742, -75.60707082465757),
        (-141.07093520913867, -75.60707082465757),
        (-141.07093520913867, -75.27429220365022),
        (-142.25471206460742, -75.27429220365022),
        (-142.25471206460742, -75.60707082465757)
    ]),
    "nickerson_2020": Polygon([
        (-137.43774034651472, -75.0620288427035),
        (-136.9364891258116, -75.0620288427035),
        (-136.9364891258116, -74.94797227760684),
        (-137.43774034651472, -74.94797227760684),
        (-137.43774034651472, -75.0620288427035)
    ])
}

# --- Simulate one processed output for visualisation ---
def simulate_sampled_data():
    print("🔄 Simulating sampled points...")
    data = []
    for class_id in [0, 1, 2]:
        lons = np.random.uniform(-137.5, -136.9, samples_per_class)
        lats = np.random.uniform(-75.07, -74.95, samples_per_class)
        data.extend([{'longitude': lon, 'latitude': lat, 'class': class_id} for lon, lat in zip(lons, lats)])
    print("✅ Simulation complete.")
    return pd.DataFrame(data)

# --- Create GeoDataFrame and Plot ---
print("🗺️ Creating GeoDataFrame and generating plot...")
df = simulate_sampled_data()
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")

# Plot and save
fig, ax = plt.subplots(figsize=(8, 8))
colors = {0: 'gray', 1: 'blue', 2: 'red'}
labels = {0: 'Background', 1: 'Lake', 2: 'Slush'}

for cls in [0, 1, 2]:
    subset = gdf[gdf['class'] == cls]
    subset.plot(ax=ax, markersize=10, label=labels[cls], color=colors[cls], alpha=0.5)

# Plot bounding box
gpd.GeoSeries(bounds_dict["hull_2018"]).plot(ax=ax, edgecolor='black', facecolor='none', linewidth=2, label="Bounding Box")

ax.set_title("Sampled Points within Bounding Box (Hull 2018)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()
plt.grid(True)
plt.savefig(visual_out)
plt.close()

print(f"✅ Preview saved to: {visual_out}")