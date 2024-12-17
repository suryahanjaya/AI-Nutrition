from tensorflow.keras.models import load_model
import os

# Create the directory if it doesn't exist
os.makedirs('saved_model', exist_ok=True)

# Load the trained model
model = load_model("best_model.keras")  # Use the final model for saving

# Save the model in SavedModel format
model.save('saved_model/food_classifier.keras')
