from tpot import TPOTClassifier
from tpot.config import classifier_config_dict
from imblearn.over_sampling import SMOTE
import joblib
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# Load preprocessed data
X_train, X_test, y_train, y_test = joblib.load('preprocessed_data.pkl')

# Apply SMOTE for class balancing
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# Create a config dictionary for tree-based models with class weights
tree_config_dict = {
    'sklearn.ensemble.ExtraTreesClassifier': {
        'n_estimators': [100],
        'criterion': ['gini', 'entropy'],
        'max_features': [0.15, 'auto'],
        'min_samples_split': [2, 3],
        'class_weight': ['balanced']  # <-- Add class weights here
    }
}

# Initialize TPOT AutoML Classifier to search for the best tree-based model with balanced class weights
tpot = TPOTClassifier(verbosity=2, generations=5, population_size=50, random_state=42, n_jobs=-1, config_dict=tree_config_dict)
tpot.fit(X_train_sm, y_train_sm)

# Export the best model pipeline code to a Python file
tpot.export('best_model_pipeline.py')

# Save the trained best model as a pickle file for future use
joblib.dump(tpot.fitted_pipeline_, 'best_model.pkl')

# Evaluate the best model on the test set
y_pred = tpot.predict(X_test)

# Metrics
print(f"Test Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, tpot.predict_proba(X_test), multi_class='ovr')}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Print the best model pipeline
print("\nBest Model Pipeline:\n")
print(tpot.fitted_pipeline_)
