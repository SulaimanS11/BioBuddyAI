import cv2
import torch
import csv
import os
from torchvision import transforms
from PIL import Image
from datetime import datetime
import time
import matplotlib.pyplot as plt

from cnn_snake import SnakeNet

# === Set up model ===
model = SnakeNet()
model.load_state_dict(torch.load("snake_model.pth", map_location="cpu"))
model.eval()

# === Transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === CSV Setup ===
os.makedirs("outputs", exist_ok=True)
csv_path = os.path.join("outputs", "detections.csv")

# Create CSV with header if not exists
if not os.path.exists(csv_path):
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Species", "Threat Level", "Response Time (s)"])

# === Webcam Capture ===
cap = cv2.VideoCapture(0)

frame_counter = 0
detections_summary = []

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tensor = transform(img_pil).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.softmax(output, dim=1)[0, 1].item()
        label = "Venomous" if prob > 0.5 else "Non-venomous"

    end_time = time.time()
    response_time = round(end_time - start_time, 3)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Log to CSV
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, label, f"{prob:.2f}", response_time])

    # Store for PNG summary
    detections_summary.append((timestamp, label, prob, response_time))

    # Display live
    cv2.putText(frame, f"{label} ({prob:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if label == "Non-venomous" else (0, 0, 255), 2)
    cv2.imshow("Snake Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# === Generate Summary PNG ===
species = [x[1] for x in detections_summary]
times = [x[0][-8:] for x in detections_summary]  # Just HH:MM:SS
probs = [x[2] for x in detections_summary]
responses = [x[3] for x in detections_summary]

plt.figure(figsize=(12, 6))
plt.plot(times, probs, label="Threat Probability", marker='o')
plt.plot(times, responses, label="Response Time (s)", marker='x')
plt.xticks(rotation=45)
plt.ylabel("Values")
plt.title("Live Detection Summary")
plt.legend()
plt.tight_layout()

summary_png_path = os.path.join("outputs", "detection_summary.png")
plt.savefig(summary_png_path)
print(f"✅ Summary PNG saved to {summary_png_path}")
