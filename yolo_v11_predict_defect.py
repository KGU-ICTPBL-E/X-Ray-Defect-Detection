import os
import cv2
from pathlib import Path
from ultralytics import YOLO

# 프로젝트 설정
RESULT_DIR = "./results/defect/20251120_092725"
TARGET_CLASSES = None
IMAGE_SIZE = 640
GPU_DEVICE = 0

# 클래스별 색상 (BGR 형식)
COLORS = [
    (0, 0, 255),    # 빨간색 (Red) - defect
]

# Validation 데이터셋 이미지 로드
import glob
VAL_IMAGE_DIR = "./DEFECT_DATASET/images/val"
test_images = sorted(glob.glob(f"{VAL_IMAGE_DIR}/*.jpg"))
print(f"Validation 이미지 개수: {len(test_images)}")

# 훈련된 YOLO 모델 로드
model = YOLO(f"{RESULT_DIR}/train/weights/best.pt")

# 이미지 추론 실행
print(f"{len(test_images)}개 이미지에 대해 추론을 실행합니다...")
results = model.predict(
    test_images,
    imgsz=IMAGE_SIZE,
    device=GPU_DEVICE
)

# 결과 저장 디렉토리 생성
predict_dir = f"{RESULT_DIR}/predict"
os.makedirs(predict_dir, exist_ok=True)

# 각 이미지의 추론 결과 처리
for i, result in enumerate(results):
    # 파일명 추출
    filename = Path(test_images[i]).stem

    # 원본 이미지 로드
    original_image = cv2.imread(test_images[i])

    # Bounding box가 있는 경우에만 처리
    if result.boxes is not None and len(result.boxes) > 0:
        # 원본 이미지에 박스 그리기
        annotated_image = original_image.copy()

        # 각 탐지된 객체의 박스 처리
        for j, box in enumerate(result.boxes):
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())

            # 지정된 클래스에 해당하는 경우만 처리
            if TARGET_CLASSES is None or class_id in TARGET_CLASSES:
                # 색상 선택
                if TARGET_CLASSES is not None:
                    color_index = TARGET_CLASSES.index(class_id)
                else:
                    color_index = class_id
                color = COLORS[color_index % len(COLORS)]

                # 박스 좌표 추출 (xyxy 형식)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # 박스 그리기
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

                # 클래스 이름과 신뢰도 표시
                label = f"defect: {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                # 라벨 배경 그리기
                cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10),
                             (x1 + label_size[0], y1), color, -1)

                # 라벨 텍스트 그리기
                cv2.putText(annotated_image, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # predict 디렉토리에 이미지 저장
        predict_output_path = f"{predict_dir}/{filename}.png"
        cv2.imwrite(predict_output_path, annotated_image)
        print(f"저장됨: {predict_output_path}")

print("\n모든 Detection 이미지 생성 및 저장이 완료되었습니다.")