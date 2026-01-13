import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

MODEL_PATH = "model/tomato_model.h5"
DATA_DIR = "data/raw/train"

st.set_page_config(page_title="Tomato Disease Detection", layout="centered")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()
class_names = sorted(os.listdir(DATA_DIR))

st.title("🍅 Tomato Plant Disease Detection")
st.write("Upload a tomato leaf image to detect disease")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((224,224))
    st.image(image, caption="Uploaded Image")

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"🌱 Disease Detected: **{predicted_class}**")
