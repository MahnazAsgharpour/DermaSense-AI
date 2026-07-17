import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from utils.dataset import MultimodalSkinDataset
from models.fusion_model import FusionModel

# --- Config ---
BATCH_SIZE = 16
EPOCHS = 10
SENSOR_DIM = 10
NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data ---
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_ds = MultimodalSkinDataset("data/train.csv","data/images","data/sensors",transform)
val_ds   = MultimodalSkinDataset("data/val.csv","data/images","data/sensors",transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# --- Model ---
model = FusionModel(sensor_dim=SENSOR_DIM, num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# --- Training Loop ---
for epoch in range(EPOCHS):
    model.train()
    for imgs, sensors, labels in train_loader:
        imgs, sensors, labels = imgs.to(DEVICE), sensors.to(DEVICE), labels.to(DEVICE)
        preds = model(imgs, sensors)
        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, sensors, labels in val_loader:
            imgs, sensors, labels = imgs.to(DEVICE), sensors.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs, sensors)
            _, pred_cls = preds.max(1)
            correct += (pred_cls == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Val Acc: {acc:.4f}")

torch.save(model.state_dict(),"best_model.pth")
print("✅ Training complete, model saved as best_model.pth")
