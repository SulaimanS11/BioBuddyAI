import cv2  # Already included — fine
from voice.voice_command import listen_command
from cam.live_detect import start_camera

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

if __name__ == "__main__":
    main()
