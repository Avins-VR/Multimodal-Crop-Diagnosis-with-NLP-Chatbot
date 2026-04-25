import os
import pandas as pd

# ========== CONFIG ==========
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
SPLIT = "train"   # change to "train" or "valid" when needed
SOIL_CSV = "train_data.csv"  # your soil feature csv file
# ============================

split_path = os.path.join(BASE_PATH, SPLIT)
soil_df = pd.read_csv(os.path.join(BASE_PATH, SOIL_CSV))

data = []

# Collect image names + labels
for label_name in os.listdir(split_path):
    label_path = os.path.join(split_path, label_name)

    if os.path.isdir(label_path):
        images_path = os.path.join(label_path, "images")

        if os.path.exists(images_path):
            for img in os.listdir(images_path):
                if img.lower().endswith((".jpg", ".jpeg", ".png")):
                    data.append({
                        "Image_Name": img,
                        "Label": label_name
                    })

img_df = pd.DataFrame(data)

# If soil rows are less, repeat them
if len(soil_df) < len(img_df):
    soil_df = pd.concat(
        [soil_df] * (len(img_df) // len(soil_df) + 1),
        ignore_index=True
    )

soil_df = soil_df.iloc[:len(img_df)].reset_index(drop=True)
img_df = img_df.reset_index(drop=True)

# Combine
final_df = pd.concat([img_df, soil_df], axis=1)

# Save
output_file = f"{SPLIT}_final_dataset.csv"
final_df.to_csv(os.path.join(BASE_PATH, output_file), index=False)

print(f"✅ {output_file} created successfully!")