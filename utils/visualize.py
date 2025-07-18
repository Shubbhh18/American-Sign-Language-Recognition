import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def print_confusion_matrix(y_true, y_pred, class_labels=None):
    if class_labels is None:
        class_labels = [str(i) for i in range(36)]  # Assuming 36 classes (0-9, A-Z)
    cmx_data = confusion_matrix(y_true, y_pred)
    df_cmx = pd.DataFrame(cmx_data, index=class_labels, columns=class_labels)
    sns.heatmap(df_cmx, annot=True, fmt='g', cmap='Greens', square=True, cbar=True, linewidths=0.5, linecolor='lightgray')
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels))

def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curves')
    plt.legend()
    plt.tight_layout()
    plt.show()