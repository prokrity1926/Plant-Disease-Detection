import tensorflow as tf

from data.plantvillage.image_loader import test_dataset

# Load the trained model
model = tf.keras.models.load_model("plant_disease_model.keras")

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_dataset)

print("\nTest Results")
print("-------------------------")
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print("Test Accuracy (%):", test_accuracy * 100)