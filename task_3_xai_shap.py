import joblib
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Load the best model pipeline
pipeline = joblib.load('best_model.pkl')

# Load the preprocessed data (raw data)
X_train, X_test, y_train, y_test = joblib.load('preprocessed_data.pkl')

# Check if the pipeline has preprocessing steps
if len(pipeline.steps) > 1:
    # If there are preprocessing steps, transform the data
    X_train_transformed = pipeline[:-1].transform(X_train)
    X_test_transformed = pipeline[:-1].transform(X_test)
else:
    # If there are no preprocessing steps, use the original data
    X_train_transformed = X_train
    X_test_transformed = X_test

# Extract the model from the pipeline
model = pipeline.steps[-1][1]  # The last step in the pipeline is the actual model

# Define your feature names (replace these with the actual feature names)
feature_names = ['age', 'sex', 'cp_1', 'cp_2', 'cp_3', 'trestbps', 'chol', 'fbs', 'restecg_0', 
                 'restecg_1', 'thalach', 'exang', 'oldpeak', 'slope_1', 'slope_2', 'ca', 
                 'thal_3', 'thal_6', 'thal_7']

# Check if the model is tree-based (for TreeExplainer)
if isinstance(model, RandomForestClassifier):
    # Use SHAP's TreeExplainer for RandomForest
    explainer = shap.TreeExplainer(model)
else:
    # Use SHAP's KernelExplainer for non-tree-based models
    explainer = shap.KernelExplainer(model.predict, X_train_transformed)

# Calculate SHAP values
shap_values = explainer.shap_values(X_test_transformed)

# SHAP summary plot with feature names
shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names)

# SHAP dependence plot for a specific feature (e.g., the 0th feature, which exists in your feature set)
shap.dependence_plot(0, shap_values[1], X_test_transformed, feature_names=feature_names)
