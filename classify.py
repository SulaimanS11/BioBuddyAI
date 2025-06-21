# classify.py
import torch
from torchvision import transforms
from PIL import Image
from cnn_model import PoisonNet

def classify_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)

    model = PoisonNet()
    model.load_state_dict(torch.load("poison_model.pth"))
    model.eval()

    output = model(image)
    _, predicted = torch.max(output.data, 1)
    return "Poisonous" if predicted.item() == 0 else "Safe"
