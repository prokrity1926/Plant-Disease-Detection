\# 🌱 Plant Disease Detection



My first Machine Learning project for detecting plant diseases from leaf images.



\## 📌 Project Overview



This project uses a deep learning model to classify plant leaf images into different disease and healthy classes.



The model is based on \*\*MobileNetV2\*\*, a pretrained convolutional neural network.



\## 🎯 Objective



The main objective of this project is to build a simple image classification system that can identify plant diseases from leaf images.



\## 🧠 Model



\- Model: MobileNetV2

\- Transfer Learning: Yes

\- Input Image Size: 224 × 224

\- Number of Classes: 38

\- Optimizer: Adam

\- Loss Function: Sparse Categorical Crossentropy

\- Epochs: 5



\## 📊 Result



The trained model achieved:



\*\*Test Accuracy: 95.49%\*\*



Test Loss: 0.1375



\## 🌿 Plant Classes



The model can classify 38 different plant/disease categories, including:



\- Apple

\- Blueberry

\- Cherry

\- Corn

\- Grape

\- Orange

\- Peach

\- Pepper

\- Potato

\- Raspberry

\- Soybean

\- Squash

\- Strawberry

\- Tomato



\## 📂 Project Structure



```text

Plant-Disease-Detection/

│

├── data/

│   ├── plantvillage/

│   └── processed/

│

├── check\_image.py

├── train\_model.py

├── test.py

├── predict.py

├── plant\_disease\_model.keras

├── test\_leaf.jpg

├── .gitignore

└── README.md

