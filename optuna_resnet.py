import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision import models
import os
import optuna  # Optuna 라이브러리
import numpy as np

# ------------------------
# 1. 데이터셋 설정
# ------------------------
dataset_dir = "/Users/vairocana/Desktop/AI/Dog_Train"  # 폴더 안에 클래스별 이미지 존재
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("✅ Using device:", device)

def train_model(trial):
    """
    Optuna가 최적의 하이퍼파라미터를 찾기 위한 목적 함수
    """

    # 1) 최적의 Batch Size 찾기
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])

    # 데이터 로더 생성 (num_workers=8, pin_memory=True 설정)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)

    # 2) ResNet50 모델 생성
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model.to(device)

    # 3) 최적의 Optimizer 찾기 (Adam vs AdamW vs SGD)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)  # 최적의 learning rate 탐색
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)  # 최적의 weight decay 탐색

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)

    # 손실 함수 정의
    criterion = nn.CrossEntropyLoss()

    # 혼합 정밀도 학습 (FP16 활용)
    scaler = torch.amp.GradScaler()

    # 4) 모델 학습 (5 Epoch만 실행, Optuna에서 빠른 탐색을 위해)
    num_epochs = 20  
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type="mps"):  # Mixed Precision Training
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

        # 5) 검증 데이터에서 성능 평가
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.autocast(device_type="mps"):  
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = 100.0 * val_correct / val_total

        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

    return val_loss  # 검증 데이터 손실값이 최소가 되는 조합을 찾음


# ------------------------
# 6. 메인 실행 (멀티프로세싱 해결)
# ------------------------
if __name__ == "__main__":

    # 데이터셋 로드 및 분할
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = torchvision.datasets.ImageFolder(root=dataset_dir, transform=transform)
    total_images = len(full_dataset)

    # Train/Validation/Test Split (70:15:15)
    train_size = int(0.7 * total_images)
    val_size = int(0.15 * total_images)
    test_size = total_images - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    num_classes = len(full_dataset.classes)
    print(f"🔹 Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")
    print("🔹 Classes:", full_dataset.classes)

    # Optuna 실행 (20회 탐색)
    study = optuna.create_study(direction="minimize")  # 최소의 val_loss를 찾는 방향
    study.optimize(train_model, n_trials=20)  # 20회 탐색

    # 최적의 하이퍼파라미터 출력
    best_params = study.best_params
    print("\n✅ 최적의 하이퍼파라미터 찾기 완료!")
    print(best_params)

    # ------------------------
    # 7. 최적의 하이퍼파라미터를 사용하여 모델 재훈련
    # ------------------------
    batch_size = best_params["batch_size"]
    learning_rate = best_params["learning_rate"]
    weight_decay = best_params["weight_decay"]
    optimizer_name = best_params["optimizer"]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model.to(device)

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)

    print("🚀 최적의 하이퍼파라미터로 모델을 재학습하세요!")
