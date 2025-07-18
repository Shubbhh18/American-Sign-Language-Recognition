import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import time
import joblib
import os

RANDOM_SEED = 42

# Create model directory if it doesn't exist
os.makedirs('model/keypoint_classifier', exist_ok=True)

# Specify each path
dataset = 'model/keypoint_classifier/keypoint.csv'
model_save_path = 'model/keypoint_classifier/xgboost_classifier.joblib'

# Set number of classes
NUM_CLASSES = 26

# Dataset reading
print("Loading dataset...")
X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))

X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Create and train XGBoost model
print("Training XGBoost model...")
start_time = time.time()

xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    num_class=NUM_CLASSES,
    random_state=RANDOM_SEED,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

# Train the model
xgb_model.fit(X_train, y_train)

# Save the model
joblib.dump(xgb_model, model_save_path)
print(f"Saved model to {model_save_path}")

# Calculate training time
training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds")

# Make predictions
print("Evaluating model...")
y_pred = xgb_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Display confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("XGBoost Confusion Matrix")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('xgboost_confusion_matrix.png')
plt.close()

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
plt.figure(figsize=(12, 6))
feature_importance = xgb_model.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1][:20]  # Top 20 features
plt.bar(range(20), feature_importance[sorted_idx])
plt.title('XGBoost Feature Importance (Top 20)')
plt.xlabel('Feature Index')
plt.ylabel('Importance Score')
plt.savefig('xgboost_feature_importance.png')
plt.close()

print("XGBoost model evaluation completed!")
print(f"Model accuracy: {accuracy:.4f}")
print(f"Training time: {training_time:.2f} seconds")
print("\nModel has been saved and is ready for real-time classification.") 