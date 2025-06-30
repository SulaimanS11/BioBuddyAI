import base64
import requests

# Correct relative path to your image
with open("snakes/default/test_snake.jpg", "rb") as img_file:
    b64 = base64.b64encode(img_file.read()).decode("utf-8")

res = requests.post(
    "http://localhost:5000/classify",
    json={"image_base64": b64}
)

print(res.json())
