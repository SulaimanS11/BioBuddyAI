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

from picamera2 import Picamera2

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
            img = Image.open(path).convert('RGB')
            vec = extract_features(img)
            human_features.append(vec)

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

# Setup picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
picam2.set_controls({"AfMode": 1})  # Enable autofocus
picam2.start()
time.sleep(2)  # Camera warm-up

detections_summary = []

# Precomputed average features
snake_avg_features = torch.stack(snake_features).mean(dim=0)
human_avg_features = torch.stack(human_features).mean(dim=0)

print("✅ Camera started. Press 'q' to quit.")

while True:
    start_time = time.time()

    frame = picam2.capture_array()  # RGB888 (640x480x3)
    if frame is None:
        print("❌ Frame capture failed.")
        break

    frame_pil = Image.fromarray(frame)
    frame_features = extract_features(frame_pil)

    sim_snake = cosine_similarity(frame_features, snake_avg_features)
    sim_human = cosine_similarity(frame_features, human_avg_features)

    adjusted_score = sim_snake - sim_human
    label = "Snake Detected" if adjusted_score > 0.7 else "No Snake Detected / Human"

    response_time = round(time.time() - start_time, 3)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, label, f"{adjusted_score:.3f}", response_time])

    detections_summary.append((timestamp, label, adjusted_score, response_time))

    # Annotate and display
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    color = (0, 0, 255) if label == "Snake Detected" else (0, 255, 0)
    cv2.putText(frame_bgr, f"{label} ({adjusted_score:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Snake & Human Detection", frame_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

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
