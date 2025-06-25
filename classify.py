# classify.py

import torch
from torchvision import transforms
from PIL import Image
from cnn_snake import SnakeNet
from quantum.real_quantum_alert import real_quantum_decision
from voice.text_to_speech import speak
import csv
from datetime import datetime

def log_detection(snake_name, verdict, threat):
    with open("output/detections.csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now(), snake_name, verdict, threat])
    print(f"Detection logged: {snake_name}, {verdict}, {threat}")

def classify_image(image_path):
    # === Load snake class names ===
    with open("snake_classes.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return f"⚠️ Error loading image: {e}"
    if image is None:
        return "⚠️ Error: Image is None. Please check the image path."

    # === Transform and prepare image ===
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    # === Load model ===
    model = SnakeNet(num_classes=len(class_names))
    model.load_state_dict(torch.load("snake_model.pth", map_location=torch.device("cpu")))
    model.eval()

    # === Predict ===
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)[0]
        predicted_idx = torch.argmax(probs).item()
        confidence = probs[predicted_idx].item()

    # === Classification result ===
    snake_name = class_names[predicted_idx]
    threat_level = real_quantum_decision(confidence)
    verdict = f"{snake_name} (confidence: {confidence:.2f})"
    speak(f"The snake is likely a {snake_name}. Threat level: {threat_level}.")
    log_detection(snake_name, confidence, threat_level)
    return f"{verdict} — {threat_level}"

    
