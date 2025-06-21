# cam/live_detect.py
import cv2
from classify import classify_image
import tempfile

def start_camera():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Camera not found")
        return

    print("Press 's' to snap and classify. Press 'q' to quit.")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow("Live Feed", frame)
        key = cv2.waitKey(1)

        if key % 256 == ord('s'):
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_img:
                img_path = temp_img.name
                cv2.imwrite(img_path, frame)
                result = classify_image(img_path)
                print(f"🔍 Classification Result: {result}")

        elif key % 256 == ord('q'):
            print("Quitting.")
            break

    cam.release()
    cv2.destroyAllWindows()
