import zipfile
import os

# Path to your ZIP file
zip_path = "/home/glene/luna/CPOM/glene/PostDoc/5DAIS/Methods/ML/GEE/Masks/Classified_NEW-20250813T095105Z-1-001.zip"



# Output directory (same as ZIP file location)
output_dir = os.path.dirname(zip_path)

# Unzip
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(output_dir)

print(f"Unzipped to: {output_dir}")
