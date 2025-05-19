# scripts/preprocess.py

import os
from pathlib import Path
from PIL import Image
import pandas as pd
from torchvision import transforms
from tqdm import tqdm

# Paths
RAW_DATA_DIR = Path("data")
PROC_DATA_DIR = Path("data_processed")
CSV_DIR      = RAW_DATA_DIR  # train_labels.csv, val_labels.csv, test_labels.csv

# Define basic transform: resize + center crop
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
])

def process_split(split_name: str):
    csv_path = CSV_DIR / f"{split_name}_labels.csv"
    df = pd.read_csv(csv_path)
    out_dir = PROC_DATA_DIR / split_name
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
        img_path = Path(row.filepath)
        # e.g. data/blue_shirt/img1.jpg → data_processed/train/blue_shirt/img1.jpg
        rel_folder = img_path.parent.name
        save_folder = out_dir / rel_folder
        save_folder.mkdir(parents=True, exist_ok=True)
        img = Image.open(img_path).convert("RGB")
        img_t = transform(img)
        img_t.save(save_folder / img_path.name)

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        process_split(split)
    print("All images processed and saved to data_processed/")
