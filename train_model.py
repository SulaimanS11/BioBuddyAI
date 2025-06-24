# train_model.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from cnn_snake import SnakeNet  # Make sure this matches your CNN definition
from tqdm import tqdm

# === Config ===
data_dir = "snakes"
batch_size = 16
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Transformations ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Dataset and Dataloader ===
dataset = datasets.ImageFolder(data_dir, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
class_names = dataset.classes  # ['Dekay brown snake', 'boa constrictor', 'copperhead']
num_classes = len(class_names)

# === Model ===
model = SnakeNet(num_classes=num_classes)
model = model.to(device)

# === Training Setup ===
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# === Training Loop ===
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

# === Save Model and Class Labels ===
torch.save(model.state_dict(), "snake_model.pth")
with open("snake_classes.txt", "w") as f:
    f.write("\n".join(class_names))

print("✅ Model trained and saved.")
print("✅ Class labels saved to snake_classes.txt.")