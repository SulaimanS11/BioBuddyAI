import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F

# Load pretrained ResNet18 once when this module is imported
resnet18 = models.resnet18(pretrained=True)
resnet18.eval()
feature_extractor = torch.nn.Sequential(*list(resnet18.children())[:-1])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def extract_features(image: Image.Image) -> torch.Tensor:
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = feature_extractor(input_tensor)
    features = features.view(features.size(0), -1)
    return features

def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
    similarity = F.cosine_similarity(v1, v2)
    return similarity.item()
