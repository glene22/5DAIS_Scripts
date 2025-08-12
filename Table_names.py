import os
import pandas as pd

# --- CONFIGURATION ---
tif_folder = "/home/glene/luna/CPOM/glene/PostDoc/5DAIS/Methods/ML/Training/train-test-data-Lancaster/test"
metadata_path = "/home/glene/luna/CPOM/glene/PostDoc/5DAIS/Methods/ML/Training/train-test-data-Lancaster/Train-Test-Data-IDs_DLR.xlsx"
output_csv = "/home/glene/luna/CPOM/glene/PostDoc/5DAIS/Methods/ML/Training/test_tif_sceneIDs.csv"

# --- LOAD TESTING SHEET ---
metadata_df = pd.read_excel(metadata_path, sheet_name='Testing')

# --- EXTRACT site/subset NAME FROM .tif FILES ---
tif_files = [f for f in os.listdir(tif_folder) if f.endswith('.tif')]
tif_prefixes = [f.split('_groundtruth')[0] for f in tif_files]

# --- BUILD TIF DATAFRAME ---
tif_df = pd.DataFrame({
    'tif_filename': tif_files,
    'site_subset': tif_prefixes
})

# --- RENAME AND MERGE ---
metadata_df = metadata_df.rename(columns={'site/subset': 'site_subset'})
merged_df = pd.merge(tif_df, metadata_df[['site_subset', 'scene ID']], on='site_subset', how='left')

# --- EXPORT ---
merged_df.to_csv(output_csv, index=False)
print(f"✅ Exported tif + scene ID list to: {output_csv}")
