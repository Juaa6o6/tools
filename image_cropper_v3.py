import os
import json
import cv2
import numpy as np
import multiprocessing  # 병렬 처리를 위한 라이브러리
from tqdm import tqdm
from PIL import Image

#----------------------
# 전처리팀 코드 image_cropper_v2에서 병렬 처리만 추가 된 코드 입니다
#----------------------

# 기본 폴더 경로 설정 (압축 풀린 폴더 및 크롭 이미지 저장 폴더)
base_folder = "/Users/vairocana/Desktop/AI/Unzipped_DOG_Validation"  # 원본 데이터 폴더
output_base_folder = "/Users/vairocana/Desktop/AI/Cropped_DOG_Validation/"  # 크롭된 이미지 저장 폴더
os.makedirs(output_base_folder, exist_ok=True)

# 지원하는 이미지 확장자
image_extensions = (".jpg", ".jpeg", ".png")

# 13개 행동(포즈) 폴더 목록
pose_folders = [
    "BODYLOWER", "BODYSCRATCH", "BODYSHAKE", "FEETUP", "FOOTUP",
    "HEADING", "LYING", "MOUNTING", "SIT", "TAILING",
    "TAILLOW", "TURN", "WALKRUN"
]

def try_find_json(json_folder, video_name):
    """
    여러 파일 이름 후보를 순회하여, video_name과 일치하는 JSON 파일을 찾습니다.
    찾으면 해당 파일의 전체 경로를 반환하고, 없으면 None을 반환합니다.
    """
    candidates = [
        f"{video_name}.json",
        f"{video_name}.mp4.json",
    ]
    if video_name.endswith(".mp4"):
        base_name = video_name[:-4]  # .mp4 제거
        candidates.append(f"{base_name}.json")
        candidates.append(f"{base_name}.mp4.json")
    for cand in candidates:
        cand_path = os.path.join(json_folder, cand)
        if os.path.exists(cand_path):
            return cand_path
    return None

def is_valid_image(file_path):
    """
    주어진 파일이 유효한 이미지인지 확인합니다.
    """
    try:
        if not os.path.exists(file_path):
            return False
        if os.path.getsize(file_path) == 0:
            return False
        img_type = Image.open(file_path).format
        if img_type not in ['JPEG', 'PNG']:
            return False
        return True
    except Exception:
        return False

def read_image_safely(img_path):
    """
    이미지를 안전하게 읽어오는 함수입니다.
    여러 방법을 시도하여 이미지를 로드합니다.
    """
    img = cv2.imread(img_path)
    if img is not None:
        return img
    if not is_valid_image(img_path):
        return None
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is not None:
        return img
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
    1. 원본 폴더([라벨]pose_folder/pose_folder)에서 JSON 파일들을 찾습니다.
    2. 원천 폴더([원천]pose_folder/pose_folder) 내 각 비디오 폴더를 반복하여,
       각 JSON 파일과 매칭되는 이미지 파일들(예: "frame_138_..." 형태)을 찾습니다.
    3. JSON의 bounding_box 정보를 기반으로 이미지를 크롭한 후,
       결과 이미지를 Cropped_DOG/pose_folder에 저장합니다.
    """
    json_folder = os.path.join(base_folder, f"[라벨]{pose_folder}", pose_folder)
    image_root = os.path.join(base_folder, f"[원천]{pose_folder}", pose_folder)
    output_folder = os.path.join(output_base_folder, pose_folder)
    os.makedirs(output_folder, exist_ok=True)

    if not os.path.isdir(image_root):
        print(f"❌ {image_root} is not a directory. Skipping {pose_folder}")
        return

    video_folders = [d for d in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, d))]
    folder_stats = {
        "total_images": 0,
        "processed_images": 0,
        "failed_reads": 0,
        "failed_crops": 0,
        "success_saves": 0
    }
    
    for video_subfolder in tqdm(video_folders, desc=f"Processing {pose_folder}"):
        video_folder_path = os.path.join(image_root, video_subfolder)
        json_path = try_find_json(json_folder, video_subfolder)
        if not json_path:
            print(f"❌ No matching JSON found for: {video_subfolder} in {json_folder}")
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"❌ Error reading JSON file: {json_path}")
            continue

        all_images = [f for f in os.listdir(video_folder_path) if f.lower().endswith(image_extensions)]
        folder_stats["total_images"] += len(all_images)
        image_map = {img: os.path.join(video_folder_path, img) for img in all_images}

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
                if w <= 0 or h <= 0:
                    print(f"❌ Invalid bounding box dimensions in {video_subfolder}, frame {frame_num}")
                    continue
            except (ValueError, TypeError):
                print(f"❌ Invalid bounding box values in {video_subfolder}, frame {frame_num}")
                continue

            target_str = f"frame_{frame_num}_"
            matching_frames = [img for img in all_images if target_str in img]
            if not matching_frames:
                continue

            for m_img in matching_frames:
                img_path = image_map[m_img]
                if not os.path.exists(img_path):
                    print(f"❌ Image file does not exist: {img_path}")
                    folder_stats["failed_reads"] += 1
                    continue

                frame_img = read_image_safely(img_path)
                if frame_img is None:
                    print(f"❌ Could not read image: {img_path}")
                    folder_stats["failed_reads"] += 1
                    continue
                
                folder_stats["processed_images"] += 1
                height, width = frame_img.shape[:2]
                if x >= width or y >= height:
                    print(f"❌ Bounding box outside image in {m_img}")
                    folder_stats["failed_crops"] += 1
                    continue
                
                x = max(0, x)
                y = max(0, y)
                w = min(w, width - x)
                h = min(h, height - y)
                try:
                    cropped = frame_img[y:y+h, x:x+w]
                    if cropped.size == 0:
                        print(f"❌ Empty crop for {m_img}")
                        folder_stats["failed_crops"] += 1
                        continue
                    out_name = f"{video_subfolder}_frame_{frame_num}_cropped.jpg"
                    out_path = os.path.join(output_folder, out_name)
                    success = cv2.imwrite(out_path, cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    if not success:
                        print(f"❌ Failed to save image: {out_path}")
                    else:
                        folder_stats["success_saves"] += 1
                except Exception as e:
                    print(f"❌ Error processing {m_img}: {str(e)}")
                    folder_stats["failed_crops"] += 1
                    continue

    print(f"📊 Stats for {pose_folder}:")
    print(f"  - Total images found: {folder_stats['total_images']}")
    print(f"  - Images processed: {folder_stats['processed_images']}")
    print(f"  - Failed to read: {folder_stats['failed_reads']}")
    print(f"  - Failed to crop: {folder_stats['failed_crops']}")
    print(f"  - Successfully saved: {folder_stats['success_saves']}")
    print(f"✅ Done with {pose_folder}.")

if __name__ == '__main__':
    # Mac에서 multiprocessing을 사용할 때 spawn 방식 대신 fork를 사용하도록 설정 (문제가 발생하면)
    try:
        multiprocessing.set_start_method('fork')
    except RuntimeError:
        pass  # 이미 설정되어 있으면 무시

    print(f"💡 OpenCV version: {cv2.__version__}")
    print(f"💡 CPU core count: {multiprocessing.cpu_count()}")
    print("💡 Starting image processing...")

    # 병렬 처리를 위해 사용할 워커(worker) 프로세스의 개수를 결정
    # `pose_folders` 리스트의 길이와 CPU 코어 개수 중 더 작은 값을 선택하여 적절한 병렬 처리 수준을 설정
    num_workers = min(len(pose_folders), multiprocessing.cpu_count())
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(process_pose_folder, pose_folders)

    print("🚀 All pose folders processed successfully!")
