# task_1_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from scipy import stats
import joblib
import ydata_profiling as pp  # AutoEDA tool

# Step 1: Load dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df = pd.read_csv(url, names=column_names)

# Step 2: AutoEDA (Ydata Profiling)
profile = pp.ProfileReport(df, title='Heart Disease Dataset - EDA', explorative=True)
profile.to_file("heart_disease_eda_report.html")  # Generates HTML EDA report

# Step 3: Data Cleaning
df.replace('?', np.nan, inplace=True)
imputer = SimpleImputer(strategy='median')
df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca', 'thal']] = imputer.fit_transform(df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca', 'thal']])

# Step 4: Outlier Detection (Optional)
z_scores = np.abs(stats.zscore(df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']]))
df_clean = df[(z_scores < 3).all(axis=1)]  # Remove rows with outliers

# Step 5: Feature Engineering
df_clean = pd.get_dummies(df_clean, columns=['cp', 'restecg', 'slope', 'thal'], drop_first=True)
X = df_clean.drop(columns='target')
y = df_clean['target']

# Step 6: Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Save the preprocessed data
joblib.dump((X_train, X_test, y_train, y_test), 'preprocessed_data.pkl')
print("Preprocessed data saved successfully.")
