import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# === Transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Load Dataset ===
dataset = datasets.ImageFolder("dataset", transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# === Model ===
model = models.efficientnet_b0(weights="DEFAULT")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# === Training ===
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(5):  # More epochs for better accuracy
    loop = tqdm(loader)
    for imgs, labels in loop:
        imgs, labels = imgs.to("cuda"), labels.to("cuda")
        preds = model(imgs)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch [{epoch+1}/5]")
        loop.set_postfix(loss=loss.item())

# === Save Fine-Tuned Model ===
torch.save(model.state_dict(), "snake_model.pth")
print("✅ Model retrained with new images.")
