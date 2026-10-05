from PIL import Image

image_path = r"data\plantvillage_images\raw\color\Apple___Apple_scab\00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG"

image = Image.open(image_path)

print("Image opened successfully!")
print("Image size:", image.size)
print("Image mode:", image.mode)