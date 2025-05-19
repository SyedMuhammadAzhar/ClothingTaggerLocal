# scripts/test_dataset.py

from src.dataset import ApparelDataset

ds = ApparelDataset(split="train")
print(f"Train set size: {len(ds)} samples")

for idx in [0, 100, 500]:
    img_tensor, (color_idx, cat_idx) = ds[idx]
    print(f"Sample {idx:3d}: img shape={tuple(img_tensor.shape)}, "
          f"color={color_idx}, category={cat_idx}")
