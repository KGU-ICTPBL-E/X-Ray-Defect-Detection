import os
import shutil
import random

random.seed(42)

SOURCE_IMAGE_DIR = "./x_ray_400_processed/bbox_only"
SOURCE_LABEL_DIR = "./x_ray_400_processed/labels"
DATASET_DIR = "./DEFECT_DATASET"

dirs_to_create = [
    f"{DATASET_DIR}/images/train",
    f"{DATASET_DIR}/images/val",
    f"{DATASET_DIR}/labels/train",
    f"{DATASET_DIR}/labels/val"
]

for dir_path in dirs_to_create:
    os.makedirs(dir_path, exist_ok=True)
    print(f"디렉토리 생성: {dir_path}")

image_files = sorted([f for f in os.listdir(SOURCE_IMAGE_DIR) if f.endswith('.jpg')])
print(f"\n전체 이미지 파일 수: {len(image_files)}")

valid_image_files = []
for img_file in image_files:
    label_file = img_file.replace('.jpg', '.txt')
    label_path = os.path.join(SOURCE_LABEL_DIR, label_file)
    if os.path.exists(label_path):
        valid_image_files.append(img_file)

print(f"레이블이 있는 유효한 이미지 수: {len(valid_image_files)}")

random.shuffle(valid_image_files)

split_idx = int(len(valid_image_files) * 0.8)
train_files = valid_image_files[:split_idx]
val_files = valid_image_files[split_idx:]

print(f"\nTrain 데이터: {len(train_files)}장")
print(f"Val 데이터: {len(val_files)}장")

print("\nTrain 데이터 복사 중...")
for img_file in train_files:
    src_img = os.path.join(SOURCE_IMAGE_DIR, img_file)
    dst_img = os.path.join(f"{DATASET_DIR}/images/train", img_file)
    shutil.copy2(src_img, dst_img)

    label_file = img_file.replace('.jpg', '.txt')
    src_label = os.path.join(SOURCE_LABEL_DIR, label_file)
    dst_label = os.path.join(f"{DATASET_DIR}/labels/train", label_file)
    shutil.copy2(src_label, dst_label)

print(f"Train 데이터 복사 완료: {len(train_files)}개")

print("\nVal 데이터 복사 중...")
for img_file in val_files:
    src_img = os.path.join(SOURCE_IMAGE_DIR, img_file)
    dst_img = os.path.join(f"{DATASET_DIR}/images/val", img_file)
    shutil.copy2(src_img, dst_img)

    label_file = img_file.replace('.jpg', '.txt')
    src_label = os.path.join(SOURCE_LABEL_DIR, label_file)
    dst_label = os.path.join(f"{DATASET_DIR}/labels/val", label_file)
    shutil.copy2(src_label, dst_label)

print(f"Val 데이터 복사 완료: {len(val_files)}개")

print("\n데이터셋 준비가 완료되었습니다!")
print(f"데이터셋 위치: {DATASET_DIR}")
