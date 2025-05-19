# scripts/evaluate.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from src.dataset import ApparelDataset, COLOR2IDX, CAT2IDX
from src.model import build_model, NUM_COLORS, NUM_CATEGORIES

def load_checkpoint(path, device):
    cp = torch.load(path, map_location=device)
    backbone, color_head, category_head = build_model(pretrained=False, freeze_backbone=False)
    backbone.load_state_dict(cp["backbone"])
    color_head.load_state_dict(cp["color_head"])
    category_head.load_state_dict(cp["category_head"])
    backbone.to(device).eval()
    color_head.to(device).eval()
    category_head.to(device).eval()
    return backbone, color_head, category_head

def evaluate(device, backbone, color_head, category_head, loader):
    all_true_c, all_pred_c = [], []
    all_true_k, all_pred_k = [], []

    with torch.no_grad():
        for imgs, (color_lbls, cat_lbls) in loader:
            imgs = imgs.to(device)
            color_lbls = color_lbls.to(device)
            cat_lbls = cat_lbls.to(device)

            feats = backbone(imgs)
            color_logits = color_head(feats)
            cat_logits   = category_head(feats)
            preds_c = color_logits.argmax(dim=1).cpu().tolist()
            preds_k = cat_logits.argmax(dim=1).cpu().tolist()

            all_true_c.extend(color_lbls.cpu().tolist())
            all_pred_c.extend(preds_c)
            all_true_k.extend(cat_lbls.cpu().tolist())
            all_pred_k.extend(preds_k)

    return all_true_c, all_pred_c, all_true_k, all_pred_k

def plot_confusion(y_true, y_pred, labels, title, out_path):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(8,6))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.savefig(out_path)
    plt.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model
    backbone, color_head, category_head = load_checkpoint("models/best.pt", device)
    # Test loader
    test_ds = ApparelDataset("test")
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    # Evaluate
    y_true_c, y_pred_c, y_true_k, y_pred_k = evaluate(device, backbone, color_head, category_head, test_loader)

    # Classification reports
    color_report = classification_report(y_true_c, y_pred_c, target_names=list(COLOR2IDX.keys()))
    cat_report   = classification_report(y_true_k, y_pred_k, target_names=list(CAT2IDX.keys()))

    print("=== Color Classification Report ===")
    print(color_report)
    print("=== Category Classification Report ===")
    print(cat_report)

    # Confusion matrices
    os.makedirs("reports", exist_ok=True)
    plot_confusion(y_true_c, y_pred_c, list(COLOR2IDX.keys()),
                   "Color Confusion Matrix", "reports/color_cm.png")
    plot_confusion(y_true_k, y_pred_k, list(CAT2IDX.keys()),
                   "Category Confusion Matrix", "reports/cat_cm.png")

    print("Confusion matrices saved to reports/*.png")
