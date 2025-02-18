import os
import cv2
from tqdm import tqdm

# 리사이즈 할 크롭되어 있는 이미지들이 있는 경로
root_dir = "/Users/Desktop/Cropped_DOG/"
output_dir = "/Users/Desktop/Resized_Cropped_DOG/"
target_size = (224, 224)  # ResNet 기본 사이즈

# 아웃풋 디렉토리 확인
os.makedirs(output_dir, exist_ok=True)

def resize_images(input_folder, output_folder):
    """ 디렉토리는 유지하면서 모든 이미지를 리사이즈. """
    for root, _, files in os.walk(input_folder):
        for file in tqdm(files, desc=f"Processing {root}"):
            if file.endswith((".jpg", ".jpeg", ".png")):
                input_path = os.path.join(root, file)

                # 폴더 구조 유지
                relative_path = os.path.relpath(input_path, input_folder)
                save_path = os.path.join(output_folder, relative_path)

                # 부모 디렉토리 확인
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                # 이미지 로드 및 리사이즈
                img = cv2.imread(input_path)
                if img is None:
                    continue  # 이미지가 읽을 수 없으면 건너뜀

                resized_img = cv2.resize(img, target_size)

                # 리사이즈된 이미지 저장
                cv2.imwrite(save_path, resized_img)

# 리사이징 함수 실행
resize_images(root_dir, output_dir)
print("All images resized and saved while maintaining folder structure!")
