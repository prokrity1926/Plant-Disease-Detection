import pandas as pd
import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load CSV files
train_df = pd.read_csv("data/processed/train.csv")
val_df = pd.read_csv("data/processed/validation.csv")
test_df = pd.read_csv("data/processed/test.csv")

# Create class names
class_names = sorted(train_df["class_name"].unique())

# Create label mapping
class_to_index = {
    class_name: index
    for index, class_name in enumerate(class_names)
}

print("Number of classes:", len(class_names))
print("Classes:")
for index, class_name in enumerate(class_names):
    print(index, "->", class_name)


def load_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    return image, label


def create_dataset(df):
    paths = df["image_path"].values
    labels = df["class_name"].map(class_to_index).values

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


train_dataset = create_dataset(train_df)
val_dataset = create_dataset(val_df)
test_dataset = create_dataset(test_df)

print("Image datasets created successfully!")
for images, labels in train_dataset.take(1):
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("First image pixel range:", images[0].numpy().min(), images[0].numpy().max())
    print("First label:", labels[0].numpy())