import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

st.set_page_config(page_title="Paws & Pixels", page_icon="🐾")

# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('cat_dog_classifier.h5')

try:
    model = load_model()
except:
    st.error("Model not found! Run 'python train_model.py' first.")
    st.stop()

st.title("🐾 Paws & Pixels")
st.write("Upload a photo to see if it's a **Cat** or a **Dog**!")

file = st.file_uploader("Choose a photo...", type=["jpg", "png", "jpeg"])

if file is not None:
    image = Image.open(file)
    st.image(image, width=300)

    # Resize to 160x160 (Must match the training size!)
    image = ImageOps.fit(image, (160, 160), Image.Resampling.LANCZOS)
    img_array = np.asarray(image) / 255.0
    img_reshape = np.expand_dims(img_array, axis=0)

    if st.button("Identify"):
        prediction = model.predict(img_reshape)
        if prediction[0][0] > 0.5:
            st.success(f"It's a **DOG**! 🐶 (Confidence: {prediction[0][0]*100:.1f}%)")
        else:
            st.success(f"It's a **CAT**! 🐱 (Confidence: {(1-prediction[0][0])*100:.1f}%)")