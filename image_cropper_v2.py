import os
import json
import cv2
import numpy as np
import multiprocessing # 병렬 처리를 위한 라이브러리고 원도우에서도 될 텐데 코드 안되면 마지막 확인해주시면 됩니다.
from tqdm import tqdm
from PIL import Image # Pillow 패키지 임포트

base_folder = "Training/Unzipped_DOG" # 압축 풀린 폴더 경로
output_base_folder = "Training/Cropped_DOG/" # 크롭 된 이미지 저장 폴더
os.makedirs(output_base_folder, exist_ok=True) # output_base_folder 경로를 생성하고, 이미 존재해도 오류를 발생시키지 않음


image_extensions = (".jpg", ".jpeg", ".png")

# 행동 폴더 13개 분류
pose_folders = [
    "BODYLOWER", "BODYSCRATCH", "BODYSHAKE", "FEETUP", "FOOTUP",
    "HEADING", "LYING", "MOUNTING", "SIT", "TAILING",
    "TAILLOW", "TURN", "WALKRUN"
]

def try_find_json(json_folder, video_name):
    """
    여러 파일 이름 돌아가며 일치하는 JSON 파일 찾음.
    찾으면 전체 경로를 반환하고, 그렇지 않으면 `None`을 반환.
    """
    # 1) exact match
    candidates = [
        f"{video_name}.json",
        f"{video_name}.mp4.json",
    ]

    # .mp4로 끝나면 제거
    if video_name.endswith(".mp4"): # .mp4로 끝나면
        base_name = video_name[:-4]  
        candidates.append(f"{base_name}.json")
        candidates.append(f"{base_name}.mp4.json")

    for cand in candidates: 
        cand_path = os.path.join(json_folder, cand)
        if os.path.exists(cand_path):
            return cand_path

    return None

def is_valid_image(file_path):
    """
    파일이 유효한 이미지인지 확인합니다.
    """
    try:
        # 파일이 존재하는지 확인
        if not os.path.exists(file_path):
            return False
            
        # 파일 크기가 0인지 확인
        if os.path.getsize(file_path) == 0:
            return False
            
        # 이미지 형식인지 확인
        img_type = Image.open(file_path).format
        if img_type not in ['JPEG', 'PNG']:
            return False
            
        return True
    except Exception:
        return False

def read_image_safely(img_path):
    """
    이미지를 안전하게 읽어옵니다. 여러 방법을 시도합니다.
    """
    # 먼저 기본 방법으로 시도
    img = cv2.imread(img_path)
    if img is not None:
        return img
        
    # 이미지가 유효한지 확인
    if not is_valid_image(img_path):
        return None
        
    # IMREAD_UNCHANGED 플래그로 시도
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is not None:
        return img
        
    # 다른 인코딩으로 시도
    try:
        with open(img_path, 'rb') as f:
            img_bytes = bytearray(f.read())
            img = cv2.imdecode(np.asarray(img_bytes), cv2.IMREAD_COLOR)
            if img is not None:
                return img
    except Exception:
        pass
        
    return None

def process_pose_folder(pose_folder):
    """
    1.  원천{pose_folder}/{pose_folder}에서 하위 폴더 찾음.
    2.  각 비디오 폴더에 대해 라벨{pose_folder}/{pose_folder}에서 JSON 파일을 매칭.
    3.  각 주석(annotation)에 대해 frame_number를 이미지 파일 'frame_{number}_…jpg'와 매칭.
    4.  잘라낸 이미지를 Cropped_DOG/pose_folder에 저장합니다.
    """
    # 예시:
    # [라벨]FOOTUP/FOOTUP/20201022_dog-footup-000023.mp4.json
    # [원천]FOOTUP/FOOTUP/20201022_dog-footup-000023.mp4/frame_138_timestamp_4600.jpg
    json_folder = os.path.join(base_folder, f"[라벨]{pose_folder}", pose_folder)
    image_root = os.path.join(base_folder, f"[원천]{pose_folder}", pose_folder)
    output_folder = os.path.join(output_base_folder, pose_folder)
    os.makedirs(output_folder, exist_ok=True)

   # 이미지 루트 내의 각 "비디오 폴더"를 반복
    # 예: "20201022_dog-footup-000023.mp4"를 하위 폴더로 사용
    if not os.path.isdir(image_root):
        print(f"❌ {image_root} is not a directory. Skipping {pose_folder}")
        return

    # 이미지 로드에 필요한 numpy 모듈 임포트
    import numpy as np

    video_folders = [d for d in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, d))]
    
    # 폴더별 처리 결과 추적
    folder_stats = {
        "total_images": 0,
        "processed_images": 0,
        "failed_reads": 0,
        "failed_crops": 0,
        "success_saves": 0
    }
    
    for video_subfolder in tqdm(video_folders, desc=f"Processing {pose_folder}"):
        video_folder_path = os.path.join(image_root, video_subfolder)

        # JSON 찾기
        json_path = try_find_json(json_folder, video_subfolder)
        if not json_path:
            # Debug print
            print(f"❌ No matching JSON found for: {video_subfolder} in {json_folder}")
            continue

        # JSON 로드
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"❌ Error reading JSON file: {json_path}")
            continue

        # 모든 이미지 가져오기
        all_images = [f for f in os.listdir(video_folder_path) if f.lower().endswith(image_extensions)]
        folder_stats["total_images"] += len(all_images)
        
        # 빠른 조회를 위한 딕셔너리 만들기
        image_map = {img: os.path.join(video_folder_path, img) for img in all_images}

        # 주석(annotation) 반복
        for ann in data.get("annotations", []):
            frame_num = ann.get("frame_number")
            if frame_num is None:
                continue
                
            bbox = ann.get("bounding_box", {})
            try:
                x = int(bbox.get("x", 0))
                y = int(bbox.get("y", 0))
                w = int(bbox.get("width", 0))
                h = int(bbox.get("height", 0))
                
                # 바운딩 박스 유효성 검사
                if w <= 0 or h <= 0:
                    print(f"❌ Invalid bounding box dimensions in {video_subfolder}, frame {frame_num}")
                    continue
            except (ValueError, TypeError):
                print(f"❌ Invalid bounding box values in {video_subfolder}, frame {frame_num}")
                continue

            # 예시 이미지 이름은 "frame_138_timestamp_4600.jpg"일 수 있음
            # frame_138_...jpg와 일치하는 이미지 찾기
            target_str = f"frame_{frame_num}_"
            matching_frames = [img for img in all_images if target_str in img]

            if not matching_frames:
                # Debug
                # print(f"No image for frame {frame_num} in {video_subfolder}")
                continue

            for m_img in matching_frames:
                img_path = image_map[m_img]
                
                # 이미지 유효성 확인 및 안전하게 읽기
                if not os.path.exists(img_path):
                    print(f"❌ Image file does not exist: {img_path}")
                    folder_stats["failed_reads"] += 1
                    continue
                    
                frame_img = read_image_safely(img_path)
                if frame_img is None:
                    print(f"❌ Could not read image (tried multiple methods): {img_path}")
                    # 파일 정보 출력
                    try:
                        print(f"  File size: {os.path.getsize(img_path)} bytes")
                        print(f"  Image type: {Image.open(img_path).format}")
                    except Exception as e:
                        print(f"  Error checking file: {str(e)}")
                    folder_stats["failed_reads"] += 1
                    continue
                
                folder_stats["processed_images"] += 1
                
                # 이미지 경계 확인
                height, width = frame_img.shape[:2]
                if x >= width or y >= height:
                    print(f"❌ Bounding box outside image in {m_img}")
                    folder_stats["failed_crops"] += 1
                    continue
                
                # 바운딩 박스가 이미지를 벗어나지 않도록 조정
                x = max(0, x)
                y = max(0, y)
                w = min(w, width - x)
                h = min(h, height - y)

                try:
                    # 크롭
                    cropped = frame_img[y:y+h, x:x+w]
                    if cropped.size == 0:
                        print(f"❌ Empty crop for {m_img}")
                        folder_stats["failed_crops"] += 1
                        continue

                    # Save
                    out_name = f"{video_subfolder}_frame_{frame_num}_cropped.jpg"
                    out_path = os.path.join(output_folder, out_name)
                    
                    # 고품질 이미지 저장 (압축률 조절)
                    success = cv2.imwrite(out_path, cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    if not success:
                        print(f"❌ Failed to save image: {out_path}")
                    else:
                        folder_stats["success_saves"] += 1
                except Exception as e:
                    print(f"❌ Error processing {m_img}: {str(e)}")
                    folder_stats["failed_crops"] += 1
                    continue

    # 폴더별 통계 출력
    print(f"📊 Stats for {pose_folder}:")
    print(f"  - Total images found: {folder_stats['total_images']}")
    print(f"  - Images processed: {folder_stats['processed_images']}")
    print(f"  - Failed to read: {folder_stats['failed_reads']}")
    print(f"  - Failed to crop: {folder_stats['failed_crops']}")
    print(f"  - Successfully saved: {folder_stats['success_saves']}")
    print(f"✅ Done with {pose_folder}.")

#윈도우에서 만약에 실행 안 되면 이부분만 바꾸시면 될거에요
if __name__ == "__main__":
    # 이미지 처리 관련 문제 발생 시 세부 정보 확인을 위한 디버그 메시지
    print(f"💡 OpenCV 버전: {cv2.__version__}")
    print(f"💡 CPU 코어 수: {multiprocessing.cpu_count()}")
    print(f"💡 이미지 처리 시작...")
    
    # 단일 쓰레드로 디버깅 모드 실행 (멀티프로세싱 문제 의심될 때)
    # for folder in pose_folders:
    #     process_pose_folder(folder)
    
    num_workers = min(len(pose_folders), multiprocessing.cpu_count())
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(process_pose_folder, pose_folders)

    print("🚀 All pose folders processed successfully!")