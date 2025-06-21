# classify.py
import torch
from torchvision import transforms
from PIL import Image
from cnn_model import PoisonNet

from quantum.real_quantum_alert import real_quantum_decision
from voice.text_to_speech import speak
import csv
from datetime import datetime

def log_detection(item_type, verdict, threat):
    with open("output/detections.csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now(), item_type, verdict, threat])

    print(f"Detection logged: {item_type}, {verdict}, {threat}")

def classify_image(image_path):
    # Transform the image
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)

    # Load the model
    model = PoisonNet()
    model.load_state_dict(torch.load("poison_model.pth"))
    model.eval()

    # Predict
    output = model(image)
    probs = torch.softmax(output, dim=1)
    confidence = probs[0][0].item()  # Confidence it's poisonous (class 0)

    # Classification verdict
    verdict = "Poisonous" if confidence > 0.5 else "Safe"

    # Quantum threat level
    threat_level = real_quantum_decision(confidence)

    # Speak out result
    speak(f"The object is {verdict}. Threat level: {threat_level}.")

    return f"{verdict} ({threat_level})"
