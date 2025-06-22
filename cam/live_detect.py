import os
import cv2
import csv
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from torchvision import transforms
from cnn_snake import SnakeNet
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# === Preprocess Frame ===
def preprocess_frame(frame):
    # Resize + Histogram Equalization (YUV)
    img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    frame_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # CLAHE on LAB
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(frame_eq, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = clahe.apply(l)
    enhanced = cv2.merge((l2, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

# === Load Model ===
model = SnakeNet()
model.load_state_dict(torch.load("snake_model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Setup CSV Logging ===
os.makedirs("outputs", exist_ok=True)
csv_path = os.path.join("outputs", "detections.csv")
if not os.path.exists(csv_path):
    with open(csv_path, mode='w', newline='') as f:
        csv.writer(f).writerow(["Timestamp", "Species", "Threat Level", "Response Time (s)"])

# === Initialize Webcam ===
cap = cv2.VideoCapture(0)
detections_summary = []

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame = preprocess_frame(frame)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    tensor = transform(img_pil).unsqueeze(0)

    with torch.no_grad():
        out = model(tensor)
        prob = torch.softmax(out, dim=1)[0, 1].item()
        theta = prob * (np.pi / 2)

        # === Quantum Probability Smoothing ===
        simulator = AerSimulator()
        qc = QuantumCircuit(1, 1)
        qc.ry(theta, 0)
        qc.measure(0, 0)
        job = simulator.run(transpile(qc, simulator))
        result = job.result()
        counts = result.get_counts(qc)
        quantum_prob = counts.get('1', 0) / sum(counts.values())

        label = "Venomous" if quantum_prob > 0.5 else "Non-venomous"
        end_time = time.time()
        response_time = round(end_time - start_time, 3)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # === Log to CSV ===
    with open(csv_path, mode='a', newline='') as f:
        csv.writer(f).writerow([timestamp, label, f"{prob:.2f}", response_time])

    detections_summary.append((timestamp, label, prob, response_time))

    # === Live Feedback ===
    color = (0, 255, 0) if label == "Non-venomous" else (0, 0, 255)
    cv2.putText(frame, f"{label} ({prob:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Snake Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# === Create PNG Summary ===
species = [x[1] for x in detections_summary]
times = [x[0][-8:] for x in detections_summary]
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

png_path = os.path.join("outputs", "detection_summary.png")
plt.savefig(png_path)
print(f"✅ Summary PNG saved to {png_path}")
