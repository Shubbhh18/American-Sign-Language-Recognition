# import torch
# import torch.nn as nn
# from torchvision import models

# class VGG16Model(nn.Module):
#     def __init__(self, num_classes):
#         super(VGG16Model, self).__init__()
#         # Load pre-trained VGG16 model
#         self.base_model = models.vgg16(weights='IMAGENET1K_V1')
        
#         # Freeze most layers for faster training
#         for param in self.base_model.features[:-4].parameters():
#             param.requires_grad = False
            
#         # Simplified classifier for faster training
#         self.base_model.classifier = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 1024),
#             nn.ReLU(True),
#             nn.Dropout(0.5),
#             nn.Linear(1024, num_classes)
#         )
        
#         # Initialize the new classifier layers
#         for m in self.base_model.classifier.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         return self.base_model(x)


import torch
import torch.nn as nn
from torchvision import models

class VGG16Model(nn.Module):
    def __init__(self, num_classes):
        super(VGG16Model, self).__init__()
        self.base_model = models.vgg16(pretrained=True)
        
        for param in list(self.base_model.features.parameters())[:-8]:
            param.requires_grad = False
            
        self.base_model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
        
        for m in self.base_model.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.base_model(x)