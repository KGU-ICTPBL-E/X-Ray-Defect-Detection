import os
from datetime import datetime
from ultralytics import YOLO

EPOCHS = 100
BATCH_SIZE = 8
IMAGE_SIZE = 640
RESULT_DIR = f"./results/defect/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
TARGET_CLASSES = None
GPU_DEVICE = 0

os.makedirs(RESULT_DIR, exist_ok=True)

# 사전 훈련된 YOLOv11 Detection 모델 사용
model = YOLO('yolo11m.pt')

# 학습 설정
train_params = {
    "data": "defect.yaml",
    "epochs": EPOCHS,
    "batch": BATCH_SIZE,
    "imgsz": IMAGE_SIZE,
    "project": RESULT_DIR,
    "val": True,
    "verbose": True,
    "device": GPU_DEVICE,
}

if TARGET_CLASSES is not None:
    train_params["classes"] = TARGET_CLASSES

results = model.train(**train_params)

model = YOLO(f"{RESULT_DIR}/train/weights/best.pt")  # 훈련된 모델 로드

# 학습 결과 평가
val_params = {
    "project": RESULT_DIR,
    "imgsz": IMAGE_SIZE,
    "verbose": True,
}

if TARGET_CLASSES is not None:
    val_params["classes"] = TARGET_CLASSES

metrics = model.val(**val_params)

# 성능 결과를 저장할 리스트
results_lines = []

# 클래스 이름 매핑 (defect.yaml 기준)
class_names = {
    0: "defect"
}

# 전체 성능 출력
print("\n=== Overall Performance ===")
results_lines.append("=== Overall Performance ===")

print(f"Box Detection mAP50: {metrics.box.map50:.4f}")
print(f"Box Detection mAP50-95: {metrics.box.map:.4f}")

# F1 Score 처리 (array일 경우 평균값 사용)
import numpy as np
f1_score = metrics.box.f1
if isinstance(f1_score, np.ndarray):
    f1_score = f1_score.mean() if len(f1_score) > 0 else 0.0
print(f"Box Detection F1 Score: {f1_score:.4f}")

results_lines.append(f"Box Detection mAP50: {metrics.box.map50:.4f}")
results_lines.append(f"Box Detection mAP50-95: {metrics.box.map:.4f}")
results_lines.append(f"Box Detection F1 Score: {f1_score:.4f}")

# 클래스별 성능 출력
print("\n=== Per-Class Detection Performance ===")
results_lines.append("\n=== Per-Class Detection Performance ===")

# 클래스별 mAP50, mAP 데이터 가져오기
box_map50_per_class = getattr(metrics.box, 'ap50', None)
box_map_per_class = getattr(metrics.box, 'ap', None)

classes_to_evaluate = TARGET_CLASSES if TARGET_CLASSES is not None else list(class_names.keys())

if box_map50_per_class is not None and len(box_map50_per_class) >= len(classes_to_evaluate):
    for i, class_id in enumerate(classes_to_evaluate):
        class_name = class_names.get(class_id, f"class_{class_id}")
        map50_val = box_map50_per_class[i] if box_map50_per_class[i] is not None else 0.0
        map_val = box_map_per_class[i] if box_map_per_class is not None and i < len(box_map_per_class) and box_map_per_class[i] is not None else 0.0

        print(f"  {class_name} (ID: {class_id})")
        print(f"    - mAP50: {map50_val:.4f}")
        print(f"    - mAP50-95: {map_val:.4f}")

        results_lines.append(f"  {class_name} (ID: {class_id})")
        results_lines.append(f"    - mAP50: {map50_val:.4f}")
        results_lines.append(f"    - mAP50-95: {map_val:.4f}")
else:
    print("No per-class detection metrics available")
    results_lines.append("No per-class detection metrics available")

# 추가 통계 정보
print("\n=== Additional Statistics ===")
results_lines.append("\n=== Additional Statistics ===")

if hasattr(metrics.box, 'mp'):
    print(f"Detection Precision: {metrics.box.mp:.4f}")
    results_lines.append(f"Detection Precision: {metrics.box.mp:.4f}")

if hasattr(metrics.box, 'mr'):
    print(f"Detection Recall: {metrics.box.mr:.4f}")
    results_lines.append(f"Detection Recall: {metrics.box.mr:.4f}")

# results.txt 파일로 저장
results_file_path = os.path.join(RESULT_DIR, "results.txt")
with open(results_file_path, 'w', encoding='utf-8') as f:
    f.write(f"YOLO v11 Defect Detection Performance Results\n")
    f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Model path: {RESULT_DIR}/train/weights/best.pt\n")
    num_classes = len(TARGET_CLASSES) if TARGET_CLASSES is not None else len(class_names)
    f.write(f"Number of classes: {num_classes}\n\n")

    for line in results_lines:
        f.write(line + '\n')

print(f"Training completed. Results saved to {RESULT_DIR}")