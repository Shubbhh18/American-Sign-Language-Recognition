import torch
import torch.nn as nn
from torchvision import models

class EfficientNetB0Model(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB0Model, self).__init__()
        self.base_model = models.efficientnet_b0(pretrained=True)
        
        in_features = self.base_model.classifier[1].in_features
        
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features=in_features, out_features=num_classes)
        )

    def forward(self, x):
        return self.base_model(x)