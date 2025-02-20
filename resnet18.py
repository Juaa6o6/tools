import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision import models
import os

# 4. ResNet18 모델 정의 부분에 레즈넷50으로 바꿀 수 있습니다(line 56).

def train_and_test():
    # ------------------------
    # 1. 경로 및 매개변수 설정
    # ------------------------
    dataset_dir = "/Users/Desktop/AI/Dog_Train"  # 각 클래스당 200장 이미지가 저장된 폴더 (하위 폴더 이름이 클래스명)
    batch_size = 32   # 데이터셋 크기가 작으므로 배치 크기 32 또는 64 정도 추천
    num_epochs = 20   # 에폭 수: 20
    learning_rate = 0.001  # 학습률: 0.001

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print("Using device:", device)

    # ------------------------
    # 2. 데이터 변환 설정 (이미지는 이미 224x224이므로 정규화만 진행)
    # ------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ------------------------
    # 3. 전체 데이터셋 로드 및 분할
    # ------------------------
    full_dataset = torchvision.datasets.ImageFolder(root=dataset_dir, transform=transform)
    total_images = len(full_dataset)  # 전체 이미지 수, 예: 2400

    # 70:15:15 분할: train, val, test
    train_size = int(0.7 * total_images)
    val_size = int(0.15 * total_images)
    test_size = total_images - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    print(f"Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    num_classes = len(full_dataset.classes)
    print("Classes:", full_dataset.classes)

    # ------------------------
    # 4. ResNet18 모델 정의
    # ------------------------
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  #model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  # ResNet50 변경시
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model.to(device)


    # ------------------------
    # 5. 손실 및 옵티마이저
    # ------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # ------------------------
    # 6. 훈련 루프
    # ------------------------
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)
        val_loss /= val_total
        val_acc = 100.0 * val_correct / val_total

        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

    # ------------------------
    # 7. 테스트
    # ------------------------
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_correct += predicted.eq(labels).sum().item()
            test_total += labels.size(0)
    test_acc = 100.0 * test_correct / test_total
    print(f"Test Accuracy: {test_acc:.2f}%")

    # ------------------------
    # 8. 모델 저장
    # ------------------------
    torch.save(model.state_dict(), "resnet18_balanced_model.pth")
    print("Training and testing complete!")

if __name__ == '__main__':
    train_and_test()
