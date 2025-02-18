import os
import json
import cv2
import multiprocessing
from tqdm import tqdm

base_folder = "/Users/Desktop/Unzipped_DOG/" # 압축 풀린 폴더 경로
output_base_folder = "/Users//Desktop/Cropped_DOG/" # 크롭 된 이미지 저장 폴더
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

def process_pose_folder(pose_folder):
    """
 	1.	원천{pose_folder}/{pose_folder}에서 하위 폴더 찾음.
	2.	각 비디오 폴더에 대해 라벨{pose_folder}/{pose_folder}에서 JSON 파일을 매칭.
	3.	각 주석(annotation)에 대해 frame_number를 이미지 파일 ‘frame_{number}_…jpg’와 매칭.
	4.	잘라낸 이미지를 Cropped_DOG/pose_folder에 저장합니다.
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

    video_folders = [d for d in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, d))]
    for video_subfolder in tqdm(video_folders, desc=f"Processing {pose_folder}"):
        video_folder_path = os.path.join(image_root, video_subfolder)

        # JSON 찾기
        json_path = try_find_json(json_folder, video_subfolder)
        if not json_path:
            # Debug print
            print(f"❌ No matching JSON found for: {video_subfolder} in {json_folder}")
            continue

        # JSON 로드
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 모든 이미지 가져오기
        all_images = [f for f in os.listdir(video_folder_path) if f.endswith(image_extensions)]
        # 빠른 조회를 위한 딕셔너리 만들기
        image_map = {img: os.path.join(video_folder_path, img) for img in all_images}

        # 주석(annotation) 반복
        for ann in data.get("annotations", []):
            frame_num = ann["frame_number"]
            bbox = ann.get("bounding_box", {})
            x, y, w, h = bbox.get("x"), bbox.get("y"), bbox.get("width"), bbox.get("height")

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
                frame_img = cv2.imread(img_path)
                if frame_img is None:
                    continue

                # 크롭
                cropped = frame_img[y:y+h, x:x+w]

                # Save
                out_name = f"{video_subfolder}_frame_{frame_num}_cropped.jpg"
                out_path = os.path.join(output_folder, out_name)
                cv2.imwrite(out_path, cropped)

    print(f"✅ Done with {pose_folder}.")

if __name__ == "__main__":
    num_workers = min(len(pose_folders), multiprocessing.cpu_count())
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(process_pose_folder, pose_folders)

    print("🚀 All pose folders processed successfully!")
