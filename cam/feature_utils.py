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

def extract_features(image):
    from torchvision.models import resnet18, ResNet18_Weights
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    try:
        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            features = model(tensor)
            if features.numel() == 0:
                print("🚫 Feature vector is empty!")
                return None
            if torch.all(features == 0):
                print("🚫 Feature vector is all zeros!")
                return None
            print("✅ Extracted feature vector mean:", features.mean().item())
            return features.squeeze()
    except Exception as e:
        print(f"❌ Exception during feature extraction: {e}")
        return None


def cosine_similarity(v1, v2):
    if not isinstance(v1, torch.Tensor):
        v1 = torch.tensor(v1, dtype=torch.float32)
    if not isinstance(v2, torch.Tensor):
        v2 = torch.tensor(v2, dtype=torch.float32)

    # Flatten if needed
    if v1.ndim > 1:
        v1 = v1.view(-1)
    if v2.ndim > 1:
        v2 = v2.view(-1)

    # Return a single scalar value
    similarity = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
    return similarity