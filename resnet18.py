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
    # 경로 및 매개변수 설정
    # ------------------------
    train_dir = "/Users/Desktop/AI/Balanced_Resized_Cropped_DOG/"  # 5000개 씩 샘플링한 데이터셋
    val_dir = "/Users/Desktop/AI/Resized_Cropped_DOG_Validation/"  # 5000개 씩 샘플링 하지 않은 데이터셋
    batch_size = 256        # 배치 크기 (원래 32 였다가 너무 오래 걸리고 레스넷18 + 램 24GB면 256도 적절하다고 나와서 바꿈)
    num_epochs = 20      #처음에 10 에폭 돌리니 너무 정확도가 낮아서 20으로 늘림
    learning_rate = 0.001       #0.0001로 했을 때 너무 느리고 한 에폭당 정확도 상승이 너무 느려서 0.001로 늘림

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu") # 맥은 MPS(GPU) 시도해보고 안되면 CPU 사용이고, 윈도우는 CUDA로 바꾸어야 합니다
    print("Using device:", device)

    # ------------------------
    # Step 2: Data Transforms
    # ------------------------
    train_transforms = transforms.Compose([
<<<<<<< HEAD
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]) #레스넷 학습을 위해서 원래 학습된 이미지넷과 맞게 정규화가 필수라고 합니다.. 어떻게 정규화 하는지는 잘 모르겠습니다
=======
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
>>>>>>> ec9de05aef06569ce0092f33f890ca22ec29250a


    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]) #레스넷 학습을 위해서 원래 학습된 이미지넷과 맞게 정규화가 필수라고 합니다.. 어떻게 정규화 하는지는 잘 모르겠습니다

    # ------------------------
    # 3. 데이터셋 및 로더 정의 (num_workers=0 컴퓨터 사항에 맞게 수정하시면 될거 같아요)
    # ------------------------
    train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=train_transforms) # 정규화된 학습 데이터셋
    val_dataset = torchvision.datasets.ImageFolder(root=val_dir, transform=val_transforms) #정규화된 검증 데이터셋

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True) 
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True) 
    # 정한 베치 사이즈에 맞게 셔플하고 학습 데이터 로드, num_workers는 CPU 병령 처리를 위한 프로세스 수(나중에는 올려서 실행해 보겠습니다), pin_memory는 메모리 고정으로 메모리 전송 속도를 높이는 것  

    num_classes = len(train_dataset.classes)
    print("Classes:", train_dataset.classes)
    #어떤 클래스들이 학습되는지

    # ------------------------
    # 4.레즈넷18 모델 정의
    # ------------------------
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) #레스넷18 모델 디폴트로 불러오기
    num_features = model.fc.in_features #모델의 특징을 가져오기
    model.fc = nn.Linear(num_features, num_classes) #마지막 레이어를 클래스 수에 맞게 13개(행동 수)로 변경
    model.to(device) #모델을 디바이스에 올리기 (제 코드는 M4 Pro GPU고 윈도우는 CUDA로 바꾸어야 합니다)

    # ------------------------
    # 5. 손실 및 옵티마이저
    # ------------------------
    criterion = nn.CrossEntropyLoss()  #분류에 주로 사용되는 손실 함수라고 합니다
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9) 
    #SGD는 배웠던 확률적 경사 하강법이고 관성(momentum)을 사용한다 정도만 알고 있습니다.
    #아담을 사용해 보라는 말도 있었는데 그냥 돌리던 코드라 그냥 SGD로 돌려보았습니다. 변경해도 될거 같습니다.

    # ------------------------
    # 6. 훈련 루프
    # ------------------------
    scaler = torch.amp.GradScaler() 
    #혼합 정밀도 학습이라고 하고 FP16(반정밀도로 16비트 사용한 데이터 형식)와 FP32(단정밀도로 32비트 사용한 데이터 형식)를 조합하여 계산을 수행하는 기법이라고 하는데 제가 이해한 바로는 .
    '''순전파와 역전파는 FP16으로 계산되는데 그 이유는 속도가 빠르고 단순 행렬 곱셈, 합성곱 등이기에 높은 정밀도가 필요 없어 충분하다고 합니다. 
    그런데 손실함수는 매우 높은 정밀도가 있어야 해서 FP32로 수행 한 손실함수입니다. 
    이렇게 혼합으로 연산을 하기에 속도와 메모리 효율성 높이고 중요한 손실함수 연산은 정확도를 높게 유지할 수 있는거 같습니다.'''

    for epoch in range(num_epochs):
        model.train() 
        '''모델을 학습 모드로 설정: 모델을 학습 모드로 설정하면 드롭아웃(과적합 방지로 랜덤 한 노드을 무시한다고 합니다. 
        예 dropout=0.5면 한 레이어에 100개 노드 중 50개만 활성화) 및 배치 정규화(각 레이어 입력값 정규화)와 같은 모듈들이 학습 모드로 설정됩니다.'''
        running_loss, correct, total = 0.0, 0, 0
        '''
        running_loss: CrossEntropyLoss 손실함수를 사용해 현재 epoch 동안 누적된 손실 값을 저장
	    correct: 예측이 정답과 일치한 샘플 수를 저장
	    total: 총 샘플 수를 저장
        accuracy = correct / total'''

        #한 배치 처리
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            #이미지와 레이블을 디바이스(GPU)에 올리기

            optimizer.zero_grad() # 파이토치 사용하면 기본적으로 각 배치에서 찾은 기울기가 누적되기 때문에 zero_grad()로 초기화해야 한다고 합니다.
            with torch.autocast(device_type="mps"):  
                outputs = model(images)
                loss = criterion(outputs, labels)
            ''' torch.autocast 클래스가 혼합정밀도 활성화합니다. 이미지를 넣어 예측값을 구하고(FP16), 손실함수를 통해 손실값을 구한다(FP32). '''

            scaler.scale(loss).backward() #손실 값을 스케일링 해서 역전파 하는데 스케일링 하는 이유는 FP16 타입은 표현 숫자 범위가 좁아 기울기가 0으로 사라지는 underflow라는 문제 발생 할 수 있기 때문. 
            scaler.step(optimizer) #옵티마이저가 가중치를 업데이트하기 전에 기울기를 다시 원래 크기로 unscale합니다.
            scaler.update() #학습 상태에 따라 스케일 팩터를 동적으로 조정합니다.
            #위 3개 함수는 혼합 정밀도 학습을 사용하면 반드시 필요하다고 합니다.

            running_loss += loss.item() * images.size(0) #한 에포크 동안 모든 배치에서 발생한 손실을 합산하려고 이렇게 계산하는 것입니다.
            _, predicted = outputs.max(1) #각 샘플에서 가장 높은 확률을 가진 클래스의 인덱스를 가져옴
            correct += predicted.eq(labels).sum().item() #예측값과 실제 정답을 비교하여 맞춘 것만 True(1), 틀린 것 False(0) 그리고 맞춘 개수 더하고 .item() 사용해서 숫자로 변환. 그러면 한 에포크 동안의 총 정답 맞춘 개수를 누적하여 저장.
            total += labels.size(0) # 한 epoch 동안 총 이미지 개수 누적

        train_loss = running_loss / total # 평균 손실
        train_acc = 100.0 * correct / total # 정확도 계산

        # ------------------------
        # 7. 검증
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
    #8. 모델 저장
    # ------------------------
    torch.save(model.state_dict(), "balanced_resnet18_mps.pth")
    print("Training Complete!")

# ------------------------
#9. 멀티프로세싱 문제 해결
# ------------------------
if __name__ == "__main__":  
    train_model()
