import os, time, csv
from datetime import datetime
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import pyttsx3
from cam.feature_utils import extract_features, cosine_similarity
from cnn_snake import SnakeNet  # Your model




##pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
##features = extract_features(pil_image, verbose=False)
def main():
    mode = input("Choose mode:\n1. Voice Command + Camera\n2. Just Image Path\n> ")

    if mode.strip() == '1':
        command = start_camera()

    elif mode.strip() == '2':
        from classify import classify_image
        image_path = input("Enter path to image: ")
        result = classify_image(image_path)
        print(f"Result: {result}")
    else:
        print("Invalid option.")
def start_camera():
    import pyttsx3

    snake_folder = "snakes"
    human_folder = "humans"
    snake_vectors = []
    human_vectors = []

    snake_metadata = []  # (vector, species name)
    for root, _, files in os.walk(snake_folder):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(root, filename)
                try:
                    img = Image.open(path).convert('RGB')
                    vec = extract_features(img)
                    if vec is not None and len(vec) == 512:
                        name = os.path.splitext(filename)[0]
                        snake_metadata.append((vec, name))
                except Exception as e:
                    print(f"❌ Snake image error: {e}")

    for root, _, files in os.walk(human_folder):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(root, filename)
                try:
                    img = Image.open(path).convert('RGB')
                    vec = extract_features(img)
                    if vec is not None and len(vec) == 512:
                        human_vectors.append(vec)
                except Exception as e:
                    print(f"❌ Human image error: {e}")

    if not snake_metadata or not human_vectors:
        print("❌ You must have at least one snake and one human image.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera could not be opened.")
        return

    engine = pyttsx3.init(driverName='nsss')
    engine.setProperty('rate', 150)

    print("✅ Camera started. Press 'q' to quit.")
    detections = []
    os.makedirs("outputs", exist_ok=True)
    csv_path = "outputs/detections.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["Timestamp", "Label", "Score", "Response Time (s)"])

    threat_history = []

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("❌ Frame read failed.")
            continue

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_vec = extract_features(frame_pil)
        if frame_vec is None:
            continue

        # Compare with preloaded snake features
        best_snake_sim = -1
        best_snake_species = "Unknown"
        for vec, name in snake_metadata:
            sim = cosine_similarity(frame_vec, vec)
            if sim > best_snake_sim:
                best_snake_sim = sim
                best_snake_species = name

        sim_human = max([cosine_similarity(frame_vec, vec) for vec in human_vectors])
        adjusted_score = best_snake_sim - sim_human

        label = f"{best_snake_species} Snake" if adjusted_score > 0.1 else "Human"
        print(f"🐍 Snake sim: {best_snake_sim:.2f}, 🧍 Human sim: {sim_human:.2f}, ➕ Score: {adjusted_score:.2f}, Label: {label}")

        if "Snake" in label:
            threat_history.append(adjusted_score)
            if len(threat_history) >= 3 and threat_history[-1] > threat_history[-2] > threat_history[-3]:
                print("⚠️ Escalating threat detected!")

        venomous_keywords = ["boa", "viper", "cobra", "rattlesnake", "copperhead"]
        if any(v in best_snake_species.lower() for v in venomous_keywords):
            warning = f"Danger! Venomous snake detected - Towards your right - {best_snake_species}, remain calm and create distance, slowly back away and no sudden movements."
            print(f"🔊 {warning}")
            engine.say(warning)
            engine.runAndWait()

        # Logging and display
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        response_time = round(time.time() - start_time, 3)
        detections.append((ts, label, adjusted_score, response_time))
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([ts, label, f"{adjusted_score:.3f}", response_time])

        color = (0, 0, 255) if "Snake" in label else (0, 255, 0)
        cv2.putText(frame, f"{label} ({adjusted_score:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Live Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Summary graph
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
