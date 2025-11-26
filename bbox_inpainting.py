import os
import glob
import numpy as np
import cv2

def get_processing_masks(img, saturation_threshold=30):
    """
    이미지에서 BBox 제거용 마스크와 BBox+Defect 제거용 마스크 두 가지를 생성하여 반환합니다.

    Args:
        img (numpy.ndarray): 입력 이미지 (BGR)
        saturation_threshold (int): 유채색(BBox)을 구분하는 채도 임계값

    Returns:
        tuple: (mask_bbox_dilated, mask_combined_dilated)
               - mask_bbox_dilated: BBox 테두리만 포함된 마스크
               - mask_combined_dilated: BBox 테두리 + 중앙 Defect 영역이 포함된 마스크
    """
    # 1. HSV 변환 및 S채널(채도) 분리
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 2. 기본 BBox 테두리 마스크 생성 (유채색 영역 탐지)
    _, mask_bbox = cv2.threshold(s, saturation_threshold, 255, cv2.THRESH_BINARY)

    # 3. BBox 내부의 Defect(중앙 1/5 영역) 마스크 생성
    mask_defect = np.zeros_like(mask_bbox)
    contours, _ = cv2.findContours(mask_bbox, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # 작은 노이즈 제거
        if cv2.contourArea(cnt) > 50:
            x, y, w, h = cv2.boundingRect(cnt)

            # 가로/세로의 1/5 크기 계산
            center_w = w // 5
            center_h = h // 5
            
            # 박스 정중앙 좌표 계산
            center_x = x + (w - center_w) // 2
            center_y = y + (h - center_h) // 2

            # Defect 마스크에 중앙 사각형 채우기
            cv2.rectangle(mask_defect, (center_x, center_y), 
                          (center_x + center_w, center_y + center_h), 255, -1)

    # 4. 마스크 결합 (BBox 테두리 + 중앙 Defect)
    mask_combined = cv2.bitwise_or(mask_bbox, mask_defect)

    # 5. 마스크 확장 (Dilation) - Inpainting 경계를 부드럽게 하기 위함
    kernel = np.ones((3, 3), np.uint8)
    mask_bbox_dilated = cv2.dilate(mask_bbox, kernel, iterations=1)
    mask_combined_dilated = cv2.dilate(mask_combined, kernel, iterations=1)

    return mask_bbox_dilated, mask_combined_dilated

def process_directory(input_dir, output_base_dir):
    """
    디렉토리 내의 이미지를 일괄 처리하여 두 가지 버전으로 저장합니다.
    """
    # 출력 경로 설정
    output_dir_bbox = os.path.join(output_base_dir, "bbox_only")
    output_dir_full = os.path.join(output_base_dir, "bbox_and_defect")

    for path in [output_dir_bbox, output_dir_full]:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")

    # 이미지 파일 탐색
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
    
    total_files = len(image_paths)
    print(f"Total images found: {total_files}")

    if total_files == 0:
        print("No images found. Please check the directory path.")
        return

    # 일괄 처리 루프
    for idx, file_path in enumerate(image_paths):
        try:
            img_array = np.fromfile(file_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                print(f"Skipping corrupted file: {file_path}")
                continue

            # 마스크 생성
            mask_bbox, mask_full = get_processing_masks(img)

            # Inpainting 수행
            # 1. BBox만 제거
            result_bbox = cv2.inpaint(img, mask_bbox, 3, cv2.INPAINT_TELEA)
            # 2. BBox + Defect 제거
            result_full = cv2.inpaint(img, mask_full, 3, cv2.INPAINT_TELEA)

            # 파일 저장 준비
            filename = os.path.basename(file_path)
            extension = os.path.splitext(filename)[1]

            # 결과 저장
            # 1. BBox Only 저장
            save_path_bbox = os.path.join(output_dir_bbox, filename)
            result, im_buf = cv2.imencode(extension, result_bbox)
            if result:
                im_buf.tofile(save_path_bbox)

            # 2. Full Processed 저장
            save_path_full = os.path.join(output_dir_full, filename)
            result, im_buf = cv2.imencode(extension, result_full)
            if result:
                im_buf.tofile(save_path_full)
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{total_files} images...")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print("Batch processing completed successfully.")

if __name__ == "__main__":
    INPUT_DIR = "Planalyze/x_ray_400"
    OUTPUT_ROOT_DIR = "Planalyze/x_ray_400_processed"

    process_directory(INPUT_DIR, OUTPUT_ROOT_DIR)