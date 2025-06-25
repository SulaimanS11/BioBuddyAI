import os, time, csv
from datetime import datetime
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

from cam.feature_utils import extract_features, cosine_similarity


def main():
    mode = input("Choose mode:\n1. Voice Command + Camera\n2. Just Image Path\n> ")

    if mode.strip() == '1':
        command = listen_command()
        print(f"You said: '{command}'")
        start_camera()

    elif mode.strip() == '2':
        from classify import classify_image
        image_path = input("Enter path to image: ")
        result = classify_image(image_path)
        print(f"Result: {result}")
    else:
        print("Invalid option.")
def start_camera():
    snake_folder = "snakes"
    human_folder = "humans"

    snake_features, human_features = [], []

    # Load known snake images
    for root, _, files in os.walk(snake_folder):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(root, filename)
                try:
                    img = Image.open(path).convert('RGB')
                    vec = extract_features(img)
                    if vec is not None and len(vec) == 512:
                        snake_features.append(vec)
                except Exception as e:
                    print(f"❌ Snake img error: {e}")

    # Load known human images
    for root, _, files in os.walk(human_folder):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(root, filename)
                try:
                    img = Image.open(path).convert('RGB')
                    vec = extract_features(img)
                    if vec is not None and len(vec) == 512:
                        human_features.append(vec)
                except Exception as e:
                    print(f"❌ Human img error: {e}")

    if not snake_features or not human_features:
        print("❌ You must have at least one snake and one human image.")
        return

    cap = cv2.VideoCapture(0)
    print("✅ Camera started. Press 'q' to quit.")

    detections = []
    os.makedirs("outputs", exist_ok=True)
    csv_path = "outputs/detections.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Label", "Score", "Response Time (s)"])

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret or frame is None:
            print("❌ Frame read failed.")
            continue

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_vec = extract_features(frame_pil)
        if frame_vec is None:
            continue

        sim_snake = max([cosine_similarity(frame_vec, vec) for vec in snake_features])
        sim_human = max([cosine_similarity(frame_vec, vec) for vec in human_features])

        adjusted_score = sim_snake - sim_human
        label = "🐍 Snake" if adjusted_score > 0.1 else "🧍 Human"

        # Log
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        response_time = round(time.time() - start_time, 3)
        detections.append((ts, label, adjusted_score, response_time))

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([ts, label, f"{adjusted_score:.3f}", response_time])

        # Show on screen
        color = (0, 0, 255) if "Snake" in label else (0, 255, 0)
        cv2.putText(frame, f"{label} ({adjusted_score:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Live Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Plot
    times = [x[0][-8:] for x in detections]
    scores = [x[2] for x in detections]
    response = [x[3] for x in detections]

    plt.figure(figsize=(12, 5))
    plt.plot(times, scores, label="Adjusted Score", marker='o')
    plt.plot(times, response, label="Response Time (s)", marker='x')
    plt.xticks(rotation=45)
    plt.title("Detection Session Summary")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/detection_summary.png")
    print("📊 Detection graph saved!")


if __name__ == "__main__":
    main()
