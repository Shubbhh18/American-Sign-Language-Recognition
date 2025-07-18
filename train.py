import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from utils.data_loader import get_dataloaders
from utils.visualize import plot_training_curves
from models.dl.cnn import SimpleCNN
from models.dl.resnet import ResNet50Model
from models.dl.vgg import VGG16Model
from models.dl.effnet import EfficientNetB0Model
import argparse
import os
import time
import numpy as np

def train_model(model, train_loader, val_loader, criterion, optimizer, model_name, num_epochs=30, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # Ensure the 'models' directory exists
    save_dir = r"D:\ASL Project\models\dl\saved_dl_models"
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
    model_save_path = os.path.join(save_dir, f"best_{model_name}.pth")
    
    # Track best model state
    best_model_state = None

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
        
        # Calculate epoch statistics
        train_loss = running_loss / total_train
        train_acc = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistics
                running_val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
        
        # Calculate validation statistics
        val_loss = running_val_loss / total_val
        val_acc = correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch results
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch:03d}/{num_epochs} | Time: {epoch_time:.2f}s | LR: {current_lr:.6f}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model (using validation accuracy as the metric)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"New best model! Val Acc: {val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break
    
    # Save the best model
    if best_model_state is not None:
        torch.save(best_model_state, model_save_path)
        print(f"Saved best model to {model_save_path} (Val Acc: {best_val_acc:.4f})")
    
    # Load the best model for final evaluation
    model.load_state_dict(best_model_state)
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)

    return model_save_path


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    parser = argparse.ArgumentParser(description="Train an ASL recognition model")
    parser.add_argument("--model_name", type=str, choices=["cnn", "resnet", "vgg", "efficientnet"], default="cnn", help="Model to train")
    parser.add_argument("--dataset_path", type=str, default=r"D:\ASL Project\dataset\American", help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset_path}")
    train_loader, val_loader, class_names = get_dataloaders(args.dataset_path, batch_size=args.batch_size)
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")

    model_options = {
        "cnn": SimpleCNN,
        "resnet": ResNet50Model,
        "vgg": VGG16Model,
        "efficientnet": EfficientNetB0Model
    }

    print(f"Initializing {args.model_name} model...")
    model_class = model_options[args.model_name]
    model = model_class(num_classes)

    # Loss function with label smoothing for better generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Optimizer with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    print(f"Starting training for {args.epochs} epochs...")
    train_model(model, train_loader, val_loader, criterion, optimizer, args.model_name, num_epochs=args.epochs)
    print("Training completed!")

# if __name__ == "__main__":
#     dataset_path = r"D:\project folders\ASL Proj DL\dataset\American"  # Update this path as needed
#     train_loader, val_loader, class_names = get_dataloaders(dataset_path)
#     num_classes = len(class_names)

#     # Dictionary to map model names to their classes
#     model_options = {
#         "cnn": SimpleCNN,
#         "resnet": ResNet50Model,
#         "vgg": VGG16Model,
#         "efficientnet": EfficientNetB0Model
#     }

#     # Choose a model by name (modify this to select your desired model)
#     model_name = "cnn"  # Change this to "resnet", "vgg", or "efficientnet" as needed
#     model_class = model_options[model_name]
#     model = model_class(num_classes)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
#     train_model(model, train_loader, val_loader, criterion, optimizer)