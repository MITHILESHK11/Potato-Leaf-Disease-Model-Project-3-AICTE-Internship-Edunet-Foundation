import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_plant_disease_model.keras")

model = load_model()

# Function to make predictions
def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    
    class_names = ['Early Blight', 'Late Blight', 'Healthy']
    return class_names[np.argmax(predictions)]

# Streamlit UI
st.set_page_config(page_title="Plant Disease Detector", layout="centered")

# Sidebar
st.sidebar.title("🌱 Plant Disease Detection")
st.sidebar.write("A system for sustainable agriculture")

# Main Interface
st.markdown("<h1 style='text-align: center; color: #2E8B57;'>🌿 Plant Disease Detection System</h1>", unsafe_allow_html=True)
st.image("Disease.jpg", use_container_width=True)

st.write("📷 Upload an image of a potato leaf, and our AI will detect if it's healthy or affected by a disease.")

# File uploader
test_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# Display Image
if test_image:
    st.image(test_image, caption="Uploaded Image", use_container_width=True)

# Predict Button
if test_image and st.button("🔍 Predict"):
    st.snow()
    prediction = model_prediction(test_image)
    st.success(f"✨ We predict it is **{prediction}**")
