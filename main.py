# main.py
import sys
from classify import classify_image

def main():
    image_path = input("Enter path to image: ")
    result = classify_image(image_path)
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
