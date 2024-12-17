# evaluate_model.py

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the trained model
model = load_model("best_model.keras")  # Use the model you want to evaluate

# Set up the test data generator (make sure you have the test images available)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    r'C:\Users\ASUS ZENBOOK\Desktop\Fast Food Classification V2\Test',  # Replace with the correct path to your test data
    target_size=(224, 224),  # Image size used for model training
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
