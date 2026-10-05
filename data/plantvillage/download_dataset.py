from datasets import load_dataset

print("Loading PlantVillage dataset...")

dataset = load_dataset(
    "mohanty/PlantVillage",
    "default"
)

print("Dataset loaded successfully!")
print(dataset)

print(dataset["train"][0])
print(dataset["train"].features)
print(dataset["train"][0]["text"])