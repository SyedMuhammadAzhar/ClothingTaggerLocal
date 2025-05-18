# ClothingTaggerLocal

AI Clothing Image Tagging Demo Project

# Add kaggle credentials

export KAGGLE_USERNAME=username
export KAGGLE_KEY=key

# List kaggle dataset and download

kaggle datasets list -s apparel-images-dataset
kaggle datasets download -d trolukovich/apparel-images-dataset

# Unzip kaggle dataset

unzip apparel-images-dataset.zip -d data/
