# src/model.py

import torch
import torch.nn as nn
from torchvision import models

NUM_COLORS = 6    # black, blue, brown, green, red, white
NUM_CATEGORIES = 5  # dress, shirt, pants, shoes, shorts

def build_model(pretrained=True, freeze_backbone=True):
    # Load a pretrained ResNet50
    backbone = models.resnet50(pretrained=pretrained)
    
    # Optionally freeze all backbone layers
    if freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False
    
    # Replace the final fc with a shared feature extractor
    num_ftrs = backbone.fc.in_features
    backbone.fc = nn.Identity()
    
    # Two separate classification heads
    color_head = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_COLORS)
    )
    category_head = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_CATEGORIES)
    )
    
    return backbone, color_head, category_head
