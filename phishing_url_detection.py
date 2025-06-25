import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
from custom_transformers import FeatureExtractor

# Load dataset
data = pd.read_csv('Dataset/phishing_site_urls.csv')

# Sanity checks
assert data['URL'].notnull().all(), "Null URLs detected"
assert data['Label'].isin(['good', 'bad']).all(), "Unexpected labels found"

# Features and target
X = data['URL']
y = data['Label'].map({'bad': 0, 'good': 1}).values  # 0 = phishing, 1 = legit

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Full pipeline
model_pipeline = Pipeline([
    ('preprocessing', FeatureExtractor()),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

# Train
model_pipeline.fit(X_train, y_train)

# Evaluate on train
y_train_pred = model_pipeline.predict(X_train)
print('--- Training Set ---')
print(f'Accuracy: {accuracy_score(y_train, y_train_pred):.4f}')
print(f'Precision: {precision_score(y_train, y_train_pred):.4f}')
print(f'Recall: {recall_score(y_train, y_train_pred):.4f}')
print(f'F1 Score: {f1_score(y_train, y_train_pred):.4f}')

# Evaluate on test
y_test_pred = model_pipeline.predict(X_test)
print('\n--- Test Set ---')
print(f'Accuracy: {accuracy_score(y_test, y_test_pred):.4f}')
print(f'Precision: {precision_score(y_test, y_test_pred):.4f}')
print(f'Recall: {recall_score(y_test, y_test_pred):.4f}')
print(f'F1 Score: {f1_score(y_test, y_test_pred):.4f}')
print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_test_pred)}')

# Save model
joblib.dump(model_pipeline, 'phishing_model.pkl')
print("\n✅ Model saved as 'phishing_model.pkl'")
