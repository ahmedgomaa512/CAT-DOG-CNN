import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# Load model
MODEL_PATH = os.path.join("model", "cat_dog_model.keras")
model = load_model(MODEL_PATH)
classes = ["Cat", "Dog"]

st.title("🐱🐶 Cat vs Dog Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB").resize((150,150))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Prepare image for model
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction, axis=1)[0]

    st.write(f"Prediction: **{classes[class_idx]}**")