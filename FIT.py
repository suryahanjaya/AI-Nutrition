import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# Load the trained model
model = load_model("best_model.keras")  # Use the best or final model

# Class labels (as an example, you should modify this to match your dataset)
class_labels = [
    "Baked Potato", "Burger", "Crispy Chicken", "Donut", "Fries", 
    "Hot Dog", "Pizza", "Sandwich", "Taco", "Taquito"
]

# Folder containing nutrition files
nutrition_folder = "nutrition_dataset"  # Ensure this matches the actual folder name

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize for model input
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array = image_array / 255.0  # Normalize
    return image_array

# Function to load nutrition data based on detected index
def load_nutrition_data(index):
    file_path = os.path.join(nutrition_folder, f"{index + 1}.txt")  # File named based on index + 1
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return file.read()
    else:
        return "No nutrition information available for this item."

# Streamlit app
st.set_page_config(page_title="FIT App", page_icon="🍔", layout="wide")
st.markdown(
    """
    <style>
    .main {background-color: #f9f9f9; font-family: 'Arial', sans-serif; padding-top: 30px;}
    .stButton>button {background-color: #ff5733; color: white; font-size: 16px; border-radius: 10px; padding: 10px 20px;}
    .stMarkdown h2 {color: #4CAF50; font-weight: bold;}
    .stMarkdown h3, .stMarkdown h4 {color: black; font-weight: bold;}  /* Updated to black text */
    .sidebar .sidebar-content {background-color: #fff8e1; padding: 20px; border-radius: 10px;}
    .stProgress>div {border-radius: 5px; background-color: #ff5733;}  /* Single color progress bar */
    .image-container {border-radius: 10px; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);}
    .info-card {border-radius: 15px; padding: 20px; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1); background-color: #074799; margin-top: 20px;}
    .info-card h3, .info-card p {font-weight: bold;}  /* Bold text in cards */
    .blue-text {color: blue; font-weight: bold;}  /* Blue text style */
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.header("FIT App Overview")
st.sidebar.markdown(
    """
    **Features:**
    - Upload food images.
    - AI-powered classification.
    - Get nutritional information instantly.

    Developed with ❤️ using Streamlit.
    """
)
st.sidebar.image("logo.jpeg", use_column_width=True)

# Welcome screen
st.title("🍴 Welcome to FIT App 🍴")
st.markdown(
    """
    ## Food Intake Tracker 

    ### Identify your favorite food and discover its nutritional information instantly!
    Upload an image of food, and our AI will classify it and provide relevant details.
    """
)


# File uploader
uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image inside a styled container
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True, clamp=True)

    # Predict the class
    st.write("Classifying...")
    progress_bar = st.progress(0)

    # Simulate progress
    for percent_complete in range(1, 101, 10):
        progress_bar.progress(percent_complete / 100)

    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)

    predicted_class_index = np.argmax(predictions)

    # Map the predicted class index to the corresponding label
    predicted_class_label = class_labels[predicted_class_index]
    
    # Display the predicted class in a more visually appealing card
    st.markdown(
        f"""
        <div class="info-card">
        <h3>Predicted Class: {predicted_class_label}</h3>
        </div>
        """, unsafe_allow_html=True)

    # Display the nutrition description in another styled card
    nutrition_info = load_nutrition_data(predicted_class_index)
    st.markdown(
        f"""
        <div class="info-card">
        <h3>Nutrition Information</h3>
        <p>{nutrition_info}</p>
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("Upload an image to begin classification.")
