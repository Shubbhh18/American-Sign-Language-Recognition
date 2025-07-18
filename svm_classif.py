import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import time
import joblib
import os

RANDOM_SEED = 42

# Create model directory if it doesn't exist
os.makedirs('model/keypoint_classifier', exist_ok=True)

# Specify each path
dataset = 'model/keypoint_classifier/keypoint.csv'
model_save_path = 'model/keypoint_classifier/svm_classifier.joblib'
scaler_save_path = 'model/keypoint_classifier/svm_scaler.joblib'

# Set number of classes
NUM_CLASSES = 26

# Dataset reading
print("Loading dataset...")
X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))

X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Scale the features for better SVM performance
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, scaler_save_path)
print(f"Saved scaler to {scaler_save_path}")

# Create and train SVM model
print("Training SVM model...")
start_time = time.time()

# Define the SVM model with optimized parameters
# Using RBF kernel which works well for this type of data
svm_model = SVC(
    C=10,
    kernel='rbf',
    gamma='scale',
    decision_function_shape='ovr',
    random_state=RANDOM_SEED,
    probability=True
)

# Train the model
svm_model.fit(X_train_scaled, y_train)

# Save the model
joblib.dump(svm_model, model_save_path)
print(f"Saved model to {model_save_path}")

# Calculate training time
training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds")

# Make predictions
print("Evaluating model...")
y_pred = svm_model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Display confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title("SVM Confusion Matrix")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('svm_confusion_matrix.png')
plt.close()

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Calculate support vectors per class
n_support = svm_model.n_support_
support_vectors_per_class = {i: n_support[i] for i in range(len(n_support))}

# Plot support vectors per class
plt.figure(figsize=(12, 6))
plt.bar(range(len(support_vectors_per_class)), list(support_vectors_per_class.values()))
plt.title('Number of Support Vectors per Class')
plt.xlabel('Class')
plt.ylabel('Number of Support Vectors')
plt.savefig('svm_support_vectors.png')
plt.close()

print("SVM model evaluation completed!")
print(f"Model accuracy: {accuracy:.4f}")
print(f"Training time: {training_time:.2f} seconds")
print(f"Total number of support vectors: {sum(n_support)}")
print("\nModel and scaler have been saved and are ready for real-time classification.") 