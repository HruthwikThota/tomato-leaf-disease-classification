
# Tomato Leaf Disease Classification Using Machine Learning

## Project Overview

This project classifies tomato leaf diseases using Traditional Machine Learning techniques.

A Linear Support Vector Machine (SVM) model is trained using HOG (Histogram of Oriented Gradients) image features.

A Streamlit web application allows users to upload tomato leaf images and predict their disease class.

## Objective

To classify tomato leaf images into 10 different categories using image processing and Traditional Machine Learning.

## Dataset

Dataset: New Plant Diseases Dataset (Augmented)

Selected classes: 10 Tomato leaf categories.

- Bacterial Spot
- Early Blight
- Late Blight
- Leaf Mold
- Septoria Leaf Spot
- Spider Mites
- Target Spot
- Tomato Yellow Leaf Curl Virus
- Tomato Mosaic Virus
- Healthy

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Scikit-image
- Pillow
- Streamlit

## Machine Learning Workflow

1. Dataset collection
2. Data exploration and visualization
3. Image preprocessing
4. Grayscale conversion
5. Image resizing to 64x64
6. HOG feature extraction
7. Label encoding
8. Feature scaling
9. Linear SVM model training
10. Hyperparameter selection
11. Model evaluation
12. Model deployment using Streamlit

## Model Details

Algorithm: Linear Support Vector Machine (SVM)

Feature Extraction: HOG

Image Size: 64x64

Number of Classes: 10

Selected C Parameter: 0.01

## Model Performance

Validation Accuracy: 56.84%

## Web Application

The Streamlit application supports:

- Tomato leaf image upload
- Image preview
- Disease prediction
- Disease descriptions
- Model validation accuracy display

## Project Structure

```text
Plant_Disease_Project/
│
├── app.py
├── requirements.txt
├── README.md
└── tomato_svm_model.joblib
