import tensorflow as tf
from tensorflow.keras import layers, models

from data.plantvillage.image_loader import (
    train_dataset,
    val_dataset,
    test_dataset,
    class_names
)


# Number of classes
NUM_CLASSES = len(class_names)

print("Number of classes:", NUM_CLASSES)


# Load pretrained MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)


# Freeze pretrained layers
base_model.trainable = False


# Build model
model = models.Sequential([
    layers.Rescaling(2.0, offset=-1.0),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(NUM_CLASSES, activation="softmax")
])


# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# Show model structure
model.summary()


# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=5
)


# Save the trained model
model.save("plant_disease_model.keras")

print("Model saved successfully!")


# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(
    test_dataset
)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)