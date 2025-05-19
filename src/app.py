from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
from torchvision import transforms
from src.model import build_model
import os

app = FastAPI()

# Load model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone, color_head, category_head = build_model(pretrained=False, freeze_backbone=False)
checkpoint = torch.load("models/best.pt", map_location=DEVICE)
backbone.load_state_dict(checkpoint["backbone"])
color_head.load_state_dict(checkpoint["color_head"])
category_head.load_state_dict(checkpoint["category_head"])
backbone.to(DEVICE).eval()
color_head.to(DEVICE).eval()
category_head.to(DEVICE).eval()

# Transforms
TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# Label maps
from src.dataset import COLOR2IDX, CAT2IDX
IDX2COLOR = {v: k for k, v in COLOR2IDX.items()}
IDX2CAT = {v: k for k, v in CAT2IDX.items()}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    img_t = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        features = backbone(img_t)
        color_logits = color_head(features)
        cat_logits = category_head(features)
        color_prob = torch.softmax(color_logits, dim=1).cpu().numpy()[0]
        cat_prob = torch.softmax(cat_logits, dim=1).cpu().numpy()[0]

    # Build response
    response = {
        "predictions": {
            "color": [{"label": IDX2COLOR[i], "confidence": float(color_prob[i])} for i in range(len(color_prob))],
            "category": [{"label": IDX2CAT[i], "confidence": float(cat_prob[i])} for i in range(len(cat_prob))]
        }
    }
    return JSONResponse(content=response)