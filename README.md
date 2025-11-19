![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU--Only-orange)
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.x-red)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

📖 Overview

This project demonstrates a complete big-data deep-learning pipeline for binary image classification (cats vs dogs) using:

Apache Spark for distributed image preprocessing

NumPy batching for memory-safe dataset handling

PyTorch for CNN model training on CPU

Scikit-learn for evaluation, metrics, and stratified splitting

The pipeline supports datasets containing 25,000+ images, even under low-memory, CPU-only virtual machines.
