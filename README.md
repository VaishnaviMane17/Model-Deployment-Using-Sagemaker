# Thyroid Disease Multi-Class Classification Model Deployment Using AWS SageMaker

This project demonstrates how to deploy a multi-class classification model for Thyroid Disease detection using AWS SageMaker. The pipeline includes data preprocessing, model training, deployment to a real-time endpoint, and making predictions via the endpoint.

---

## 🔍 Project Overview

This project classifies patients into multiple thyroid-related categories using clinical and demographic features. The model is deployed using AWS SageMaker, enabling scalable and production-ready inference.

To understand the working of the model, feature selection, data preprocessing techniques, and interpretation of predictions, **refer to the published research paper associated with this project**.

---

## 📁 Project Structure and File Functionality

| File / Folder          | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `main.py`              | Main orchestration script that uploads data, trains the model, and deploys to SageMaker. |
| `train_model.py`       | Contains the custom training logic used by the SageMaker estimator.         |
| `predict.py`           | Sends input data to the deployed endpoint and returns predictions.          |
| `utils.py`             | Utility functions for data upload and other helper functions.               |
| `requirements.txt`     | List of required Python packages.                                           |
| `data/`                | Folder containing training and testing datasets (`train-V-1.csv`, `test-V-1.csv`). |
| `model/`               | Folder to store trained model artifacts (optional if handled in SageMaker). |
| `README.md`            | This file — instructions to run the project.                                |

---
