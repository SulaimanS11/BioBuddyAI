import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class SnakeNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SnakeNet, self).__init__()
        # Load pretrained EfficientNet-B0 backbone
        weights = EfficientNet_B0_Weights.DEFAULT
        self.base_model = efficientnet_b0(weights=weights)
        
        # Replace classifier head to match number of classes
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.base_model(x)
