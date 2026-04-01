import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

st.title("🐱 Cat vs Dog Classifier")

@st.cache_resource
def load_my_model():
    return load_model('my_model.keras')

model = load_my_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocessing (adjust size if needed)
    img = image.resize((150, 150))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        st.success("🐶 Dog")
    else:
        st.success("🐱 Cat")
else:
    st.warning("Please upload an image")