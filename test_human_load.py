import os
from PIL import Image
from cam.feature_utils import extract_features
import torch

human_folder = "humans"
human_features = []

for root, _, files in os.walk(human_folder):
    for filename in files:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(root, filename)
            print("🖼 Found human image:", path)
            try:
                img = Image.open(path).convert('RGB')
                vec = extract_features(img)
                
                if vec is not None:
                    mean_val = vec.mean().item()
                    print(f"✅ Extracted feature vector mean: {mean_val:.8f}")
                    
                    # Loosened threshold: allow lower-quality inputs
                    if torch.all(vec == 0) or abs(mean_val) < 1e-6:
                        print("⚠️ Likely low-info vector (accepted for Pi Cam testing).")
                    human_features.append(vec)
                else:
                    print("⚠️ No vector returned.")
            except Exception as e:
                print(f"❌ Error processing {path}: {e}")

print(f"✅ Total good human vectors: {len(human_features)}")
