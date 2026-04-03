import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

# Set page configuration
st.set_page_config(page_title="New Cat vs Dog Classifier", layout="centered")

# Title
st.title("🐱 New Cat vs Dog Classifier 🐶")
st.write("Upload an image to classify whether it contains a cat or a dog!")

# Load model (with caching to improve performance)
@st.cache_resource
def load_trained_model():
    return load_model('model/my_model.keras')

model = load_trained_model()
class_names = ['Cat', 'Dog']

# Image uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        image_display = Image.open(uploaded_file)
        st.image(image_display, use_column_width=True)

    with col2:
        st.subheader("Prediction")

        # Make prediction
        try:
            # Prepare image for model
            img = Image.open(uploaded_file).convert('RGB')
            img = img.resize((64, 64))  # Resize to model input size
            img_array = np.array(img) / 255.0  # Normalize to 0-1
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Predict
            pred = model.predict(img_array, verbose=0)
            confidence = float(pred[0][0])

            # Get class
            class_idx = int(confidence > 0.5)
            predicted_class = class_names[class_idx]

            # Display result
            st.success(f"**Predicted: {predicted_class}**")

            # Show confidence
            if class_idx == 1:  # Dog
                dog_confidence = float(confidence * 100)
                cat_confidence = float((1 - confidence) * 100)
            else:  # Cat
                cat_confidence = float((1 - confidence) * 100)
                dog_confidence = float(confidence * 100)

            st.metric("Cat Confidence", f"{cat_confidence:.2f}%")
            st.metric("Dog Confidence", f"{dog_confidence:.2f}%")

            # Progress bar
            st.write("Prediction Confidence:")
            st.progress(float(max(cat_confidence, dog_confidence) / 100))

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
