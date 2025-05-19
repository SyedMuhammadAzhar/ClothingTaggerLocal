# ClothingTaggerLocal

AI Clothing Image Tagging Demo Project

## Project Overview

This demo implements a simplified version of an AI-driven apparel tagger (similar to Catecut). You can upload an apparel image and get predictions for **color** and **category**.

## Repository Structure

```
ClothingTaggerLocal/
├── .venv/                    # Python 3.11 virtual environment
├── data/                     # Raw Kaggle dataset (unzipped)
│   ├── black_dress/ ...      # color_category folders with images
│   └── ...
├── data_processed/           # Preprocessed images (224×224) split by train/val/test
├── models/                   # Saved model checkpoint and history
├── reports/                  # Evaluation reports & confusion matrices
├── scripts/                  # CLI scripts
│   ├── preprocess.py         # Resize & save images
│   ├── train.py              # Train & fine‑tune model
│   ├── test_dataset.py       # Validate Dataset loading
│   └── evaluate.py           # Evaluate on test set
├── src/                      # Core Python modules
│   ├── __init__.py
│   ├── app.py                # FastAPI inference endpoint
│   ├── model.py              # ResNet50 two‑head model definition
│   ├── dataset.py            # PyTorch Dataset for apparel images
│   ├── utils.py              # (optional helpers)
├── ui/                       # Streamlit frontend
│   └── streamlit_app.py
├── notebooks/                # EDA & visualization notebooks
│   └── 01_data_exploration.ipynb
├── requirements.txt          # Python dependencies
└── .gitignore
```

## Setup & Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/yourusername/ClothingTaggerLocal.git
   cd ClothingTaggerLocal
   ```

2. **Create & activate Python 3.11 venv**

   ```bash
   python3.11 -m venv .venv
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Dataset Preparation

1. **Download Kaggle dataset** (requires Kaggle credentials in env vars)

   ```bash
   export KAGGLE_USERNAME=<username>
   export KAGGLE_KEY=<key>
   kaggle datasets download -d trolukovich/apparel-images-dataset
   unzip apparel-images-dataset.zip -d data/
   ```

2. **Data Exploration & Splitting**

   - Open `notebooks/01_data_exploration.ipynb` to explore and generate `data/{train,val,test}_labels.csv`.
   - Or run in CLI:

     ```bash
     python - <<EOF
     # custom script or commands from the notebook
     EOF
     ```

3. **Preprocess images** (resize, crop)

   ```bash
   python scripts/preprocess.py
   ```

   This outputs `data_processed/train`, `data_processed/val`, `data_processed/test`.

## Training the Model

Train in two phases (heads, then fine‑tune backbone):

```bash
python -m scripts.train \
  --batch-size 32 \
  --lr 1e-3 \
  --epochs-head 5 \
  --epochs-finetune 3 \
  --output-dir models
```

- **models/best.pt**: best checkpoint
- **models/history.csv**: training/validation metrics

## Evaluation

```bash
pip install seaborn        # if not already installed
python -m scripts.evaluate
```

- **reports/color_cm.png** and **reports/cat_cm.png**: confusion matrices
- Console outputs: precision, recall, F1 per class

## 🔌 Inference API & Frontend

1. **Start FastAPI server**

   ```bash
   uvicorn src.app:app --reload
   ```

2. **Launch Streamlit UI**

   ```bash
   streamlit run ui/streamlit_app.py
   ```

3. **Usage**

   - Upload a `.jpg/.png` image of apparel
   - Click **Predict**
   - View top color & category predictions with confidences

## Next Steps & Extensions

- Add Dockerfiles for reproducible deployment
- Integrate logging/monitoring for model drift
- Extend UI with batch uploads or image gallery
- Deploy API/UI to cloud (AWS/GCP/Azure free tiers)

## References

- Kaggle Apparel Images Dataset: [https://www.kaggle.com/datasets/trolukovich/apparel-images-dataset](https://www.kaggle.com/datasets/trolukovich/apparel-images-dataset)
- PyTorch Transfer Learning: [https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

---

_Demo project by \[Syed Muhammad Azhar]_
