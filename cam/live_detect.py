import os
import time
import csv
from datetime import datetime

import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from cam.feature_utils import extract_features, cosine_similarity
from cnn_snake import SnakeNet  # Your model



# === Setup ===

snake_folder = "snakes"
human_folder = "humans"

snake_features = []
for root, _, files in os.walk(snake_folder):
    for filename in files:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(root, filename)
            img = Image.open(path).convert('RGB')
            vec = extract_features(img)
            snake_features.append(vec)

human_features = []
for root, _, files in os.walk(human_folder):
    for filename in files:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(root, filename)
            print("🖼 Found human image:", path)  # 👈 Debug print here
            try:
                img = Image.open(path).convert('RGB')
                vec = extract_features(img)
                if vec is not None and len(vec) == 512:
                    human_features.append(vec)
                else:
                    print(f"⚠️ Invalid feature vector from: {path}")
            except Exception as e:
                print(f"❌ Error processing {path}: {e}")

            print(f"✅ Loaded {len(human_features)} human feature vectors.")


# Load SnakeNet model
model = SnakeNet()
model.load_state_dict(torch.load("snake_model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Output folder and CSV
os.makedirs("outputs", exist_ok=True)
csv_path = os.path.join("outputs", "detections.csv")

if not os.path.exists(csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Label", "Adjusted Score", "Response Time (s)"])


detections_summary = []

print("✅ Camera started. Press 'q' to quit.")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open webcam.")
    exit()

threat_history = []

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ Frame capture failed.")
        break

    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame_features = extract_features(frame_pil)

    sim_snake = cosine_similarity(frame_features, snake_features)
    sim_human = cosine_similarity(frame_features, human_features)
    adjusted_score = sim_snake - sim_human

    label = "No Snake Detected"
    threat = "None"

    if adjusted_score > 0.7:
        # Save frame to temp file
        temp_path = "outputs/temp_frame.jpg"
        frame_pil.save(temp_path)

        # Classify snake and get threat level
        from classify import classify_image, quantum_decision
        species, confidence = classify_image(temp_path)
        threat = quantum_decision(species, confidence)

        label = f"{species} ({confidence:.2f}) - {threat}"
    else:
        label = "No Snake Detected"

    # Display result
    color = (0, 0, 255) if "THREAT" in threat else (0, 255, 0)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imshow("Snake Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    threat_history.append(confidence)

    if len(threat_history) >= 3 and threat_history[-1] > threat_history[-2] > threat_history[-3]:
        print("⚠️ Escalating threat detected!")

    else:
        label = "No Snake Detected"



cv2.destroyAllWindows()

# Plot summary
times = [x[0][-8:] for x in detections_summary]
scores = [x[2] for x in detections_summary]
responses = [x[3] for x in detections_summary]

plt.figure(figsize=(12, 6))
plt.plot(times, scores, label="Adjusted Score", marker='o')
plt.plot(times, responses, label="Response Time (s)", marker='x')
plt.xticks(rotation=45)
plt.ylabel("Values")
plt.title("Snake Detection Summary")
plt.legend()
plt.tight_layout()

summary_png_path = os.path.join("outputs", "detection_summary.png")
plt.savefig(summary_png_path)
print(f"✅ Summary PNG saved to {summary_png_path}")
