import os
import cv2
from tqdm import tqdm

# 리사이즈 할 크롭되어 있는 이미지들이 있는 경로
root_dir = "/Users/vairocana/Desktop/AI/Cropped_DOG/"
output_dir = "/Users/vairocana/Desktop/AI/Resized_Cropped_DOG/"
target_size = (224, 224)  # ResNet standard input size

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

def resize_images(input_folder, output_folder):
    """Recursively resizes all images in a folder while keeping the directory structure."""
    for root, _, files in os.walk(input_folder):
        for file in tqdm(files, desc=f"Processing {root}"):
            if file.endswith((".jpg", ".jpeg", ".png")):
                input_path = os.path.join(root, file)

                # Preserve the folder structure
                relative_path = os.path.relpath(input_path, input_folder)
                save_path = os.path.join(output_folder, relative_path)

                # Ensure the parent directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                # Load and resize image
                img = cv2.imread(input_path)
                if img is None:
                    continue  # Skip if image is unreadable

                resized_img = cv2.resize(img, target_size)

                # Save resized image
                cv2.imwrite(save_path, resized_img)

# Run the resizing function
resize_images(root_dir, output_dir)
print("🚀 All images resized and saved while maintaining folder structure!")
