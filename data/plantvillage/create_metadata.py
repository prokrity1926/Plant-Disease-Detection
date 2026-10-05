from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/plantvillage_images/raw/color")

rows = []

for class_dir in sorted(DATA_DIR.iterdir()):
    if class_dir.is_dir():
        for image_path in class_dir.iterdir():
            if image_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                rows.append({
                    "image_path": str(image_path),
                    "class_name": class_dir.name
                })

df = pd.DataFrame(rows)

output_path = "data/processed/metadata.csv"
df.to_csv(output_path, index=False)

print("Metadata created successfully!")
print("Total images:", len(df))
print("Total classes:", df["class_name"].nunique())
print("Saved to:", output_path)

print("\nFirst 5 rows:")
print(df.head())