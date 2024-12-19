import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load a pre-trained model (e.g., MobileNetV2 for food image classification)
@st.cache_resource
def load_model():
    model = tf.keras.applications.MobileNetV2(weights="imagenet")
    return model

# Map food items to their calorie values (offline database)
FOOD_CALORIE_DATABASE = {
    "apple": 52,
    "banana": 89,
    "pizza": 266,
    "burger": 295,
    "sushi": 200,
    "fried chicken": 246,
    "salad": 152,
}

def predict_food(image, model):
    """Process the image and predict food items using the pre-trained model."""
    # Resize the image to match the input size of the model
    image = image.resize((224, 224))  # MobileNetV2 input size
    image_array = np.array(image)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)

    # Predict using the model
    predictions = model.predict(image_array)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)
    return decoded_predictions[0]  # Top 3 predictions

def calculate_calories(predictions, calorie_db):
    """Match predictions to calorie database and calculate total calories."""
    total_calories = 0
    details = []

    for pred in predictions:
        label = pred[1]  # Predicted label (e.g., 'apple')
        confidence = pred[2]  # Confidence score
        calories = calorie_db.get(label, None)  # Lookup calories in the database

        if calories:
            details.append(f"{label.capitalize()} ({confidence * 100:.1f}% confidence) - {calories} kcal")
            total_calories += calories

    return details, total_calories

# Streamlit app
st.set_page_config(page_title="AI Nutritionist (Local)")
st.header("AI Nutritionist App (Offline)")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Load model
model = load_model()

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Analyze button
    if st.button("Analyze Food & Calculate Calories"):
        # Predict food items
        predictions = predict_food(image, model)

        # Match predictions with calorie database and calculate
        details, total_calories = calculate_calories(predictions, FOOD_CALORIE_DATABASE)

        # Display results
        st.subheader("Nutrition Analysis:")
        if details:
            st.write("\n".join(details))
            st.write(f"**Total Calories:** {total_calories} kcal")
        else:
            st.write("No recognizable food items found in the database.")
