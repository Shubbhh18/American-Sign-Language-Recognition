import torch
from utils.data_loader import get_dataloaders
from utils.visualize import print_confusion_matrix
from models.dl.cnn import SimpleCNN
from models.dl.resnet import ResNet50Model
from models.dl.vgg import VGG16Model
from models.dl.effnet import EfficientNetB0Model
import argparse
import os

def evaluate_model(model, val_loader, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    print_confusion_matrix(all_labels, all_preds, class_labels=class_names)


if __name__ == "__main__":
    # Command-line arguments for flexibility
    parser = argparse.ArgumentParser(description="Evaluate an ASL recognition model")
    parser.add_argument("--model_name", type=str, choices=["cnn", "resnet", "vgg", "efficientnet"], default="cnn", help="Model to evaluate")
    parser.add_argument("--dataset_path", type=str, default=r"D:\ASL Project\dataset\American", help="Path to dataset")
    args = parser.parse_args()

    # Load validation data
    _, val_loader, class_names = get_dataloaders(args.dataset_path)
    num_classes = len(class_names)

    # Dictionary to map model names to their classes
    model_options = {
        "cnn": SimpleCNN,
        "resnet": ResNet50Model,
        "vgg": VGG16Model,
        "efficientnet": EfficientNetB0Model
    }

    # Select and instantiate the model
    model_name = args.model_name
    model_class = model_options[model_name]
    model = model_class(num_classes)

    # Dynamic .pth file loading
    model_save_path = f"models/saved_dl_models/best_{model_name}.pth"
    if not os.path.exists(model_save_path):
        raise FileNotFoundError(f"Model file {model_save_path} not found. Please train the {model_name} model first.")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    print(f"Loaded model from {model_save_path}")

    # Evaluate the model
    evaluate_model(model, val_loader, class_names)

if __name__ == "__main__":
    dataset_path = r"D:\ASL Project\dataset\American"  # Update this path as needed
    _, val_loader, class_names = get_dataloaders(dataset_path)
    num_classes = len(class_names)

    # Load the best model (uncomment the same model used in train.py)
    model = SimpleCNN(num_classes)
    # model = ResNet50Model(num_classes)
    # model = VGG16Model(num_classes)
    # model = EfficientNetB0Model(num_classes)
    model.load_state_dict(torch.load("models/saved_dl_models/best_model.pth"))

    evaluate_model(model, val_loader, class_names)