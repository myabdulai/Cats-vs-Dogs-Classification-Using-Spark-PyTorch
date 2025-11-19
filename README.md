![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU--Only-orange)
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.x-red)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

📌 README.md — CNN Image Classification with Spark Preprocessing
🐱🐶 CNN-Based Cats vs. Dogs Classifier (Spark + PyTorch)

This project implements a binary image classification system for distinguishing between cats and dogs using a hybrid big-data + deep learning pipeline. Apache Spark is used for distributed preprocessing of over 25,000 images, and PyTorch is used for memory-efficient CNN training using streamed NumPy batches on a CPU-only environment.
The project demonstrates how to process large image datasets under hardware constraints while maintaining model accuracy and reproducibility.

🚀 Project Features

✔ Distributed image preprocessing with Apache Spark
✔ Memory-safe image batching using .npy files
✔ Custom PyTorch CNN trained with streaming DataLoader
✔ Stratified train/test split
✔ Training curves and model evaluation metrics
✔ Confusion matrix, ROC curve, and classification report
✔ Runs on CPU-only machines (e.g., low-RAM VMs)

📁 Project Structure
.
├── spark_preprocess_images.py     # Spark-based image preprocessing
├── train_cnn_streamed.py          # Memory-efficient PyTorch CNN training
├── cnn_dataset_stream.py          # Streaming dataset loader
├── X_part_*.npy                   # Image batches
├── y_part_*.npy                   # Label batches
├── accuracy_curve.png
├── loss_curve.png
├── roc_curve.png
├── confusion_matrix.png
└── README.md


🔧 Technologies Used
Big Data & Preprocessing

Apache Spark

PyArrow

NumPy

Deep Learning

PyTorch

Torchvision

scikit-learn

Visualization

Matplotlib

Seaborn

🧹 1. Dataset Preprocessing with Spark

Large image datasets cannot fit into memory, so this project preprocesses images using Spark:

Loads images from directory (Cat/, Dog/)

Resizes to 64×64

Normalizes pixel intensities

Saves them into 13 small .npy files

X_part_0.npy ... X_part_12.npy

y_part_0.npy ... y_part_12.npy

Run preprocessing:

python3 spark_preprocess_images.py

🧠 2. Streaming CNN Training (PyTorch)

The CNN uses:

2 convolution layers

ReLU activation

Max pooling

Dropout

Fully connected classifier

Sigmoid output layer (binary classification)

Train the streamed model:

python3 train_cnn_streamed.py


The training script automatically:

Loads .npy batches from disk

Builds DataLoader objects

Performs stratified sampling

Trains a lightweight CNN

Evaluates on test subset

Prints:

Accuracy

Classification report

Confusion matrix

ROC curve

Loss and accuracy curves

📊 3. Sample Outputs
✔ Training Accuracy Curve

✔ Training Loss Curve

✔ ROC Curve

✔ Confusion Matrix

🧪 Model Performance Summary

The streamed CNN successfully learns from batched data

Achieves high accuracy on balanced test splits

Handles >25,000 images using minimal memory

Demonstrates the feasibility of running CNN training on CPU-only systems

⚠️ Challenges Faced

Arrow serialization failures in Spark

Memory termination ("Killed") during full dataset loading

CPU-only training limitations

Need for stratified splitting to avoid single-class test sets

Slow preprocessing on low-resource VMs

🔮 Future Improvements

Move training to GPU environments (Colab, AWS, university HPC)

Use transfer learning models (ResNet, MobileNet)

Add data augmentation to reduce overfitting

Add hyperparameter tuning

Deploy CNN as an API (FastAPI / Flask)

Store processed data in HDFS for scalable retraining

📝 How to Clone This Repository

Replace yourusername with your GitHub username:

git clone https://github.com/yourusername/cats-dogs-spark-cnn.git
cd cats-dogs-spark-cnn

📜 License

MIT License — free to use, modify, and distribute.

🤝 Contributions

Contributions, pull requests, and improvements are welcome!

📧 Author

Mohammed Yushawu Abdulai
Graduate Student — Health Informatics
Michigan Technological University
