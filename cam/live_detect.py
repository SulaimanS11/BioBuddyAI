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
            img = Image.open(path).convert('RGB')
            vec = extract_features(img)

            if vec is not None and len(vec) == 512:
                human_features.append(vec)
            else:
                print(f"⚠️ Skipped bad image: {path}")

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

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        print("❌ Frame capture failed.")
        break

    if frame is None:
        print("❌ Frame capture failed.")
        break

    frame_pil = Image.fromarray(frame)
    frame_features = extract_features(frame_pil)

    if not human_features:
        print("❌ No valid human features found. Skipping frame.")
        continue  # Skip this frame if no valid human data

    sims = [cosine_similarity(frame_features, vec) for vec in snake_features]
    max_sim = max(sims)
    best_match_index = sims.index(max_sim)

    sim_human = cosine_similarity(frame_features, human_features)

    adjusted_score = max_sim - sim_human
    is_snake = adjusted_score > 0.7

    # Load class names (once only)
    if 'class_names' not in globals():
        with open("snake_classes.txt", "r") as f:
            class_names = [line.strip() for line in f.readlines()]

    label = f"{class_names[best_match_index]}" if is_snake else "No Snake Detected / Human"

    # Optionally call quantum threat assessment
    from quantum.real_quantum_alert import real_quantum_decision
    from voice.text_to_speech import speak

    if is_snake:
        threat = real_quantum_decision(max_sim)
        speak(f"{label} detected. Threat level: {threat}.")

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
