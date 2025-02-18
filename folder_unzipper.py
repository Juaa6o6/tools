import os
import zipfile

# 데이터 경로 설정
data_root = "반려동물 구분을 위한 동물 영상/Training/DOG"  # Training 데이터 경로
output_folder = "Unzipped_DOG"  # 압축 해제된 데이터 저장 폴더. 경로는 코드 실행 위치 기준

# 폴더 생성
os.makedirs(output_folder, exist_ok=True)

# ZIP 파일들 자동 해제
for file in os.listdir(data_root):
    if file.endswith(".zip"):
        file_path = os.path.join(data_root, file)
        extract_folder = os.path.join(output_folder, file.replace(".zip", ""))

        # ZIP 파일 해제
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
        
        print(f"압축 해제 완료: {file} → {extract_folder}") 


