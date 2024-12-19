import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained model
model = load_model("best_model.keras")  # Use the best or final model

# Class labels (as an example, you should modify this to match your dataset)
class_labels = [
    "Baked Potato", "Burger", "Crispy Chicken", "Donut", "Fries", 
    "Hot Dog", "Pizza", "Sandwich", "Taco", "Taquito"
]

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize for model input
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array = image_array / 255.0  # Normalize
    return image_array

# Streamlit app
st.title("Food Classification App")
uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict
    st.write("Classifying...")
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions)

    # Map the predicted class index to the corresponding label
    predicted_class_label = class_labels[predicted_class_index]
    
    # Display the predicted class
    st.write(f"Predicted Class: {predicted_class_label}")
