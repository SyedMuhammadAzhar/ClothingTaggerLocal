# scripts/train.py

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import pandas as pd

from src.dataset import ApparelDataset
from src.model import build_model

def train_epoch(backbone, color_head, category_head, loader, optimizers, criterions, device):
    backbone.train(); color_head.train(); category_head.train()
    running_loss = 0.0
    correct_color, correct_cat, total = 0, 0, 0

    for imgs, (color_lbls, cat_lbls) in tqdm(loader, desc="Train"):
        imgs = imgs.to(device)
        color_lbls = color_lbls.to(device)
        cat_lbls = cat_lbls.to(device)

        # Forward
        features = backbone(imgs)
        color_logits = color_head(features)
        cat_logits   = category_head(features)
        loss_color   = criterions[0](color_logits, color_lbls)
        loss_cat     = criterions[1](cat_logits, cat_lbls)
        loss = loss_color + loss_cat

        # Backward
        optimizers[0].zero_grad(); optimizers[1].zero_grad(); optimizers[2].zero_grad()
        loss.backward()
        for opt in optimizers:
            opt.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds_color = torch.max(color_logits, 1)
        _, preds_cat   = torch.max(cat_logits, 1)
        correct_color += (preds_color == color_lbls).sum().item()
        correct_cat   += (preds_cat == cat_lbls).sum().item()
        total += imgs.size(0)

    avg_loss = running_loss / total
    acc_color = correct_color / total
    acc_cat   = correct_cat   / total
    return avg_loss, acc_color, acc_cat

def eval_epoch(backbone, color_head, category_head, loader, criterions, device):
    backbone.eval(); color_head.eval(); category_head.eval()
    running_loss = 0.0
    correct_color, correct_cat, total = 0, 0, 0

    with torch.no_grad():
        for imgs, (color_lbls, cat_lbls) in tqdm(loader, desc="Eval"):
            imgs = imgs.to(device)
            color_lbls = color_lbls.to(device)
            cat_lbls = cat_lbls.to(device)

            features = backbone(imgs)
            color_logits = color_head(features)
            cat_logits   = category_head(features)
            loss = criterions[0](color_logits, color_lbls) + criterions[1](cat_logits, cat_lbls)

            running_loss += loss.item() * imgs.size(0)
            _, preds_color = torch.max(color_logits, 1)
            _, preds_cat   = torch.max(cat_logits, 1)
            correct_color += (preds_color == color_lbls).sum().item()
            correct_cat   += (preds_cat == cat_lbls).sum().item()
            total += imgs.size(0)

    avg_loss = running_loss / total
    acc_color = correct_color / total
    acc_cat   = correct_cat   / total
    return avg_loss, acc_color, acc_cat

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets & Loaders
    train_ds = ApparelDataset("train")
    val_ds   = ApparelDataset("val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    backbone, color_head, category_head = build_model(pretrained=True, freeze_backbone=True)
    backbone.to(device); color_head.to(device); category_head.to(device)

    # Optimizers: separate for backbone (for later unfreeze) and heads
    opt_backbone   = torch.optim.AdamW(backbone.parameters(),   lr=args.lr*0.1)
    opt_color_head = torch.optim.AdamW(color_head.parameters(), lr=args.lr)
    opt_cat_head   = torch.optim.AdamW(category_head.parameters(), lr=args.lr)

    # Loss functions
    criterion_color = nn.CrossEntropyLoss()
    criterion_cat   = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    history = []

    # Phase 1: train heads only
    for epoch in range(args.epochs_head):
        train_stats = train_epoch(backbone, color_head, category_head, train_loader,
                                  [opt_backbone, opt_color_head, opt_cat_head],
                                  [criterion_color, criterion_cat], device)
        val_stats = eval_epoch(backbone, color_head, category_head, val_loader,
                               [criterion_color, criterion_cat], device)

        print(f"[Head Epoch {epoch+1}/{args.epochs_head}] "
              f"Train Loss {train_stats[0]:.4f} | Val Loss {val_stats[0]:.4f} "
              f"| Train Acc (C,C) {train_stats[1]:.3f},{train_stats[2]:.3f} "
              f"| Val Acc (C,C) {val_stats[1]:.3f},{val_stats[2]:.3f}")

        history.append({
            "phase":"head", "epoch":epoch+1,
            "train_loss":train_stats[0], "val_loss":val_stats[0],
            "train_acc_color":train_stats[1], "train_acc_cat":train_stats[2],
            "val_acc_color":val_stats[1], "val_acc_cat":val_stats[2],
        })

    # Phase 2: unfreeze backbone and fine‑tune all layers
    for param in backbone.parameters():
        param.requires_grad = True

    for epoch in range(args.epochs_finetune):
        train_stats = train_epoch(backbone, color_head, category_head, train_loader,
                                  [opt_backbone, opt_color_head, opt_cat_head],
                                  [criterion_color, criterion_cat], device)
        val_stats = eval_epoch(backbone, color_head, category_head, val_loader,
                               [criterion_color, criterion_cat], device)

        print(f"[FT Epoch {epoch+1}/{args.epochs_finetune}] "
              f"Train Loss {train_stats[0]:.4f} | Val Loss {val_stats[0]:.4f} "
              f"| Train Acc (C,C) {train_stats[1]:.3f},{train_stats[2]:.3f} "
              f"| Val Acc (C,C) {val_stats[1]:.3f},{val_stats[2]:.3f}")

        history.append({
            "phase":"finetune", "epoch":epoch+1,
            "train_loss":train_stats[0], "val_loss":val_stats[0],
            "train_acc_color":train_stats[1], "train_acc_cat":train_stats[2],
            "val_acc_color":val_stats[1], "val_acc_cat":val_stats[2],
        })

        # Save best
        if val_stats[0] < best_val_loss:
            best_val_loss = val_stats[0]
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save({
                "backbone": backbone.state_dict(),
                "color_head": color_head.state_dict(),
                "category_head": category_head.state_dict(),
            }, os.path.join(args.output_dir, "best.pt"))

    # Save training history
    os.makedirs(args.output_dir, exist_ok=True)
    pd.DataFrame(history).to_csv(os.path.join(args.output_dir, "history.csv"), index=False)
    print("Training complete. Models & history saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train apparel tagger")
    parser.add_argument("--batch-size",    type=int,   default=32)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--epochs-head",   type=int,   default=5)
    parser.add_argument("--epochs-finetune", type=int, default=3)
    parser.add_argument("--output-dir",    type=str,   default="models")
    args = parser.parse_args()
    main(args)
