![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU--Only-orange)
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.x-red)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

🐶🐱 Cats vs Dogs Classification Using Spark + PyTorch
A Distributed Preprocessing + Memory-Efficient CNN Training Pipeline
📖 Overview

This project demonstrates a complete big-data deep-learning pipeline for binary image classification (cats vs dogs) using:

Apache Spark for distributed image preprocessing

NumPy batching for memory-safe dataset handling

PyTorch for CNN model training on CPU

Scikit-learn for evaluation, metrics, and stratified splitting

The pipeline supports datasets containing 25,000+ images, even under low-memory, CPU-only virtual machines.

🧰 Key Features

✔ Distributed preprocessing across 2 virtual machines
✔ Full dataset batching into .npy chunks
✔ Memory-efficient CNN training using streaming
✔ Balanced, stratified train–test split
✔ Confusion matrix, ROC curve, accuracy curve, loss curve
✔ Suitable for big-data environments or constrained hardware

📂 Dataset

Kaggle Dataset:
https://www.kaggle.com/datasets/salader/dogsvscats

Total images: ~25,000

Classes: Cat (0) and Dog (1)

🖥️ Technologies & Libraries
Purpose	Tools
Distributed preprocessing	Apache Spark, Hadoop
Batch creation	NumPy
CNN Training	PyTorch
Evaluation	scikit-learn
Visualization	Matplotlib
⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/yourusername/cats-dogs-spark-cnn.git
cd cats-dogs-spark-cnn

2️⃣ Install Python dependencies
pip install torch torchvision numpy matplotlib scikit-learn

3️⃣ Organize dataset

Place Kaggle dataset here:

data/raw/PetImages/Cat
data/raw/PetImages/Dog

4️⃣ Run Spark preprocessing
python spark_preprocess_images.py

5️⃣ Train the CNN (streaming-based)
python train_cnn_streamed.py

🧠 CNN Architecture
Conv2D (32 filters, 3×3)
ReLU
MaxPool2D (2×2)

Conv2D (64 filters, 3×3)
ReLU
MaxPool2D (2×2)

Flatten
Fully Connected (128)
ReLU
Output Layer (Sigmoid)


Optimized for CPU training, not GPU.

📊 Results
Metric	Value
Test Accuracy	~0.78
AUC	~0.88
📉 Visualizations
![Accuracy Curve](C:/Users/amysh/Desktop/HI/12_2025 Fall Semester/Big Data Analytics_SAT5165/Small project 4/accuracy_curve.png)
![Loss Curve](C:/Users/amysh/Desktop/HI/12_2025 Fall Semester/Big Data Analytics_SAT5165/Small project 4/loss_curve.png)
![ROC Curve](C:/Users/amysh/Desktop/HI/12_2025 Fall Semester/Big Data Analytics_SAT5165/Small project 4/roc_curve.png)
![Confusion Matrix](C:/Users/amysh/Desktop/HI/12_2025 Fall Semester/Big Data Analytics_SAT5165/Small project 4/confusion_matrix.png)
