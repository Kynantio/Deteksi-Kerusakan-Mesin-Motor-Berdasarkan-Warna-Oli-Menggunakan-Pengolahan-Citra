# src/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from model import ResNet50Transfer

# Path dataset
DATA_DIR = "../dataset"
MODEL_PATH = "../models/best_model.pth"

# Hyperparameters
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_CLASSES = 4

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformasi gambar
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Dataset dan DataLoader
full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

print("Mapping kelas yang digunakan:")
print(full_dataset.class_to_idx)

class_names = full_dataset.classes

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_set, val_set = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

# Model, loss, optimizer
model = ResNet50Transfer(num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"[{epoch+1}/{EPOCHS}] Train Loss: {avg_loss:.4f}")

    # Validasi
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f"Validation Accuracy: {acc:.2f}%")

# Simpan model
os.makedirs("../models", exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print("✅ Model saved to", MODEL_PATH)
