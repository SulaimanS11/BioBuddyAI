from cnn_snake import SnakeNet
import torch

model = SnakeNet()

# Save initialized model weights to file
torch.save(model.state_dict(), "snake_model.pth")
print("✅ Dummy model saved as snake_model.pth")
