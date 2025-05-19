# src/dataset.py

import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Map string labels to numeric classes
COLOR2IDX = {"black": 0, "blue":1, "brown":2, "green":3, "red":4, "white":5}
CAT2IDX   = {"dress":0, "shirt":1, "pants":2, "shoes":3, "shorts":4}

# Shared transforms (you can add augmentations here)
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])

EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])

class ApparelDataset(Dataset):
    def __init__(self, split: str, processed_data_dir="data_processed", csv_dir="data"):
        """
        split: “train”, “val”, or “test”
        """
        csv_path = Path(csv_dir) / f"{split}_labels.csv"
        self.df = pd.read_csv(csv_path)
        self.base_dir = Path(processed_data_dir) / split
        self.transforms = TRAIN_TRANSFORMS if split=="train" else EVAL_TRANSFORMS

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # path in processed folder
        img_path = self.base_dir / f"{row.color}_{row.category}" / Path(row.filepath).name
        img = Image.open(img_path).convert("RGB")
        img = self.transforms(img)

        color_idx = COLOR2IDX[row.color]
        cat_idx   = CAT2IDX[row.category]
        return img, (color_idx, cat_idx)
