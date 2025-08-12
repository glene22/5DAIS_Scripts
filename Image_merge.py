import os
import rasterio
import numpy as np

mask_folder = "/home/glene/luna/CPOM/glene/PostDoc/5DAIS/Methods/ML/Training/Thresholded/tm_crevasse"

lake_files = [f for f in os.listdir(mask_folder) if f.lower().startswith("modified_lakemask")]
slush_files = [f for f in os.listdir(mask_folder) if "slushmask" in f.lower()]

def extract_region_year(filename):
    parts = filename.split("_")
    region = parts[2]
    year = parts[3].split(".")[0]
    return region, year

paired_masks = []
for lake_file in lake_files:
    region, year = extract_region_year(lake_file)
    match = [f for f in slush_files if region in f and year in f]
    if match:
        paired_masks.append((lake_file, match[0]))
    else:
        print(f"⚠️ No slush mask found for {lake_file}")

for lake_file, slush_file in paired_masks:
    lake_path = os.path.join(mask_folder, lake_file)
    slush_path = os.path.join(mask_folder, slush_file)

    region, year = extract_region_year(lake_file)
    output_path = os.path.join(mask_folder, f"merged_LakeSlushMask_{region}_{year}.tif")

    with rasterio.open(lake_path) as lake_src:
        lake_data = lake_src.read(1)
        meta = lake_src.meta.copy()
        shape = lake_data.shape
        merged = np.zeros(shape, dtype=np.uint8)  # 0 = background

        # Assign lake pixels (value = 1)
        merged[lake_data > 0] = 1

    with rasterio.open(slush_path) as slush_src:
        slush_data = slush_src.read(1)
        if slush_data.shape != shape:
            raise ValueError(f"Shape mismatch between {lake_file} and {slush_file}")
        
        # Assign slush pixels (value = 2) — overwrites lakes
        merged[slush_data > 0] = 2

    meta.update(dtype=rasterio.uint8, count=1)

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(merged, 1)

    # Optional print counts
    unique, counts = np.unique(merged, return_counts=True)
    print(f"✅ {output_path}: {dict(zip(unique, counts))}")
