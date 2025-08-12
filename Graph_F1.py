import pandas as pd
import matplotlib.pyplot as plt
import os

# Path to CSV
csv_path = "5DAIS/Methods/ML/Training/Testing_data/Binary_RF_Eval_With_PrecisionRecall.csv"
df = pd.read_csv(csv_path)

# Create a shorter version of the raster name for x-axis labels
df["raster_short"] = df["raster"].str.replace("/", "_", regex=False)

# Sort by manual F1 score for visual clarity
df = df.sort_values("f1_manual", ascending=False).reset_index(drop=True)

# Set up plot
plt.figure(figsize=(14, 6))
bar_width = 0.4
x = range(len(df))

# Plot F1 scores
plt.bar(x, df["f1_manual"], width=bar_width, label="F1 Manual", color="grey")
plt.bar([i + bar_width for i in x], df["f1_thresh"], width=bar_width, label="F1 Thresholded", color="lightblue")

# X-axis labels
plt.xticks([i + bar_width / 2 for i in x], df["raster_short"], rotation=90, fontsize=8)

# Axis labels and title
plt.xlabel("Raster")
plt.ylabel("F1 Score")
plt.title("F1 Score per Raster: Manual vs. Thresholded")

# Force y-axis from 0 to 1
plt.ylim(0, 1)

# Add legend and layout
plt.legend()
plt.tight_layout()

# Save to same folder as CSV
output_folder = os.path.dirname(csv_path)
output_path = os.path.join(output_folder, "F1_scores_all_rasters.png")
plt.savefig(output_path, dpi=300)
plt.show()
