import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
import os

def train_model():
    # ------------------------
    # Step 1: Set Paths & Params
    # ------------------------
    train_dir = "/Users/Desktop/AI/Balanced_Resized_Cropped_DOG/"  # Balanced dataset
    val_dir = "/Users/Desktop/AI/Resized_Cropped_DOG_Validation/"  # Validation dataset
    batch_size = 256
    num_epochs = 20
    learning_rate = 0.001

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print("Using device:", device)

    # ------------------------
    # Step 2: Data Transforms
    # ------------------------
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ------------------------
    # Step 3: Datasets & Loaders (Fix `num_workers=0`)
    # ------------------------
    train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=train_transforms)
    val_dataset = torchvision.datasets.ImageFolder(root=val_dir, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    num_classes = len(train_dataset.classes)
    print("Classes:", train_dataset.classes)

    # ------------------------
    # Step 4: Define ResNet18 Model
    # ------------------------
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model.to(device)

    # ------------------------
    # Step 5: Loss & Optimizer
    # ------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # ------------------------
    # Step 6: Training Loop (Fixed GradScaler)
    # ------------------------
    scaler = torch.amp.GradScaler()  # ✅ FIXED: Removed `cuda.amp`

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type="mps"):  # Mixed Precision for Speed
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        # ------------------------
        # Step 7: Validation
        # ------------------------
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                with torch.autocast(device_type="mps"):  # Mixed Precision
                    val_outputs = model(val_images)
                    v_loss = criterion(val_outputs, val_labels)

                val_loss += v_loss.item() * val_images.size(0)
                _, v_predicted = val_outputs.max(1)
                val_correct += v_predicted.eq(val_labels).sum().item()
                val_total += val_labels.size(0)

        val_loss /= val_total
        val_acc = 100.0 * val_correct / val_total

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

    # ------------------------
    # Step 8: Save Model
    # ------------------------
    torch.save(model.state_dict(), "balanced_resnet18_mps.pth")
    print("Training Complete!")

# ------------------------
# Step 9: Fix Multiprocessing Issue
# ------------------------
if __name__ == "__main__":  # ✅ FIXED: Proper multiprocessing safeguard
    train_model()
