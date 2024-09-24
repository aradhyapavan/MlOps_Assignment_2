
# Heart Disease Prediction API

This repository contains the code and documentation for the **Heart Disease Prediction Model**, developed as part of **MLOps Assignment 2**. The project covers the full machine learning lifecycle, from data preprocessing and model training to explainable AI and deployment on the cloud.

### Live API
The trained model is deployed and accessible as a live API:
- **[Heart Disease Prediction API](https://mlops-assignment-2-group-56-09a0941178fc.herokuapp.com/)**

---

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Project Structure](#project-structure)
6. [Model Details](#model-details)
7. [API Endpoints](#api-endpoints)
8. [Deployment on Heroku](#deployment-on-heroku)
9. [Technologies Used](#technologies-used)

---

## Overview

This project demonstrates the development of a machine learning model to predict heart disease based on patient clinical data. The project includes:

- **Data Preprocessing**: Cleaning and transforming the Heart Disease dataset from the UCI repository.
- **Model Training**: Automating model selection and hyperparameter tuning using **TPOT AutoML**.
- **Explainable AI (XAI)**: Providing insights into model predictions using **SHAP**.
- **Cloud Deployment**: Deploying the model as a web API using **Heroku**.

---

## Features

- **Data Preprocessing**: Handles missing values, outliers, one-hot encodes categorical variables, and scales numerical features.
- **Machine Learning Model**: Uses **ExtraTreesClassifier** selected via AutoML for heart disease prediction.
- **Explainability**: SHAP is used to visualize feature importance and understand the model’s decisions.
- **Live API**: The trained model is deployed as a RESTful API for real-time predictions.
- **Cloud Deployment**: Hosted on Heroku, ensuring scalability and ease of access.

---

## Installation

To run the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/aradhyapavan/MlOps_Assignment_2.git
   cd MlOps_Assignment_2
   ```

2. **Install the dependencies**:
   Make sure you have Python 3.x installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask API locally**:
   ```bash
   python app.py
   ```

4. **Access the API locally**:
   The API will be accessible at:
   ```bash
   http://localhost:5000/
   ```

---

## Usage

The Heart Disease Prediction API accepts input features via a POST request and returns a prediction of heart disease severity (0-4). The features include clinical and medical data, such as **age**, **cholesterol levels**, and **maximum heart rate**.

### Example cURL Request:
```bash
curl -X POST https://mlops-assignment-2-group-56-09a0941178fc.herokuapp.com/predict \
-H "Content-Type: application/json" \
-d '{
  "features": [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 1, 0, 1, 0, 1, 0, 0, 1]
}'
```

#### Example Response:
```json
{
  "prediction": 1
}
```

This response indicates a prediction of **Class 1**, meaning the presence of mild heart disease.

---

## Project Structure

The repository is structured as follows:

```
MlOps_Assignment_2/
├── app.py              # Main Flask API for predictions
├── task_1_preprocessing.py  # Data preprocessing script
├── task_2_training.py       # Model training using TPOT
├── task_3_xai_shap.py       # Explainable AI using SHAP
├── requirements.txt    # Dependencies required for the project
├── preprocessed_data.pkl  # Preprocessed data for model training
├── best_model.pkl      # Trained model saved as a pickle file
├── Procfile            # Heroku configuration file
└── README.md           # Project description and instructions
```

---

## Model Details

### **Dataset**: 
- The **Heart Disease dataset** was sourced from the UCI repository and contains 303 records with 14 attributes, such as age, cholesterol levels, and chest pain type.

### **Model**: 
- The best-performing model selected by **TPOT AutoML** was the **ExtraTreesClassifier**. This model was trained using balanced class weights to handle class imbalance and fine-tuned through hyperparameter optimization.

### **Metrics**:
- **Accuracy**: 53% on the test set.
- **ROC-AUC Score**: 0.67.
- **Precision & Recall**: Computed for each class (0-4), indicating different severities of heart disease.

---

## API Endpoints

### **Home Route** (`/`)
- Provides a summary of available features and usage instructions.

### **Prediction Route** (`/predict`)
- **Method**: POST
- **Input**: A JSON object with 18 input features representing patient clinical data.
- **Output**: A prediction ranging from **0-4** indicating the severity of heart disease.

---

## Deployment on Heroku

The model was deployed on **Heroku** using a Flask API. The deployment process involved:

1. **Creating the Flask API** to load the model and handle predictions.
2. **Pushing the code to GitHub** and connecting the repository to Heroku for automatic deployment.
3. **Defining environment settings** and dependencies via `Procfile` and `requirements.txt`.
4. **Monitoring** the application via the Heroku dashboard.

You can access the deployed API at:
- **[Heart Disease Prediction API](https://mlops-assignment-2-group-56-09a0941178fc.herokuapp.com/)**

---

## Technologies Used

- **Python 3.x**
- **Flask**: For building the web API.
- **TPOT AutoML**: For automated model selection and hyperparameter tuning.
- **scikit-learn**: For machine learning algorithms and preprocessing.
- **SMOTE**: To handle class imbalance.
- **SHAP**: For model interpretability and feature importance explanations.
- **Heroku**: Cloud platform for deploying the Flask API.


---

### Contributors

- **Aradhya Pavan H S** (2022ac05457)
- **Akash S** (2022ac05316)
- **Apurba Roy** (2022ac05075)
- **Gayathry S** (2022ac05263)
- **Jesinta Rozario** (2022ac05566)



