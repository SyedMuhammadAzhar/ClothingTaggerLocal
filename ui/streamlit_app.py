import streamlit as st
import requests
from PIL import Image
import io

API_URL = "http://localhost:8000/predict/"

st.title("Apparel Tagger Demo")
uploaded_file = st.file_uploader("Choose an image of clothing…", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Predict"):
        with st.spinner("Predicting…"):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            resp = requests.post(API_URL, files=files)
        if resp.status_code == 200:
            data = resp.json()["predictions"]
            st.subheader("Color Predictions")
            for item in data["color"]:
                st.write(f"{item['label']}: {item['confidence']:.2f}")
            st.subheader("Category Predictions")
            for item in data["category"]:
                st.write(f"{item['label']}: {item['confidence']:.2f}")
        else:
            st.error("Prediction failed.")