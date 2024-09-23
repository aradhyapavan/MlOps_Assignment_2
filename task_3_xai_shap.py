# task_3_xai.py

import joblib
import shap

# Load the best model
model = joblib.load('best_model.pkl')

# Load the preprocessed data
X_train, X_test, y_train, y_test = joblib.load('preprocessed_data.pkl')

# Explainability using SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# SHAP summary plot
shap.summary_plot(shap_values, X_test)

# SHAP dependence plot for an individual feature
shap.dependence_plot(0, shap_values[1], X_test)
