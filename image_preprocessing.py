import cv2
import numpy as np
import logging


def remove_background(cropped_image):
# GrabCut 알고리즘을 사용하여 이미 잘라낸 알약 이미지에서 배경을 제거
    
    try:
        h, w = cropped_image.shape[:2]
        if h < 10 or w < 10: return cropped_image, np.zeros((h, w), np.uint8)
        mask = np.zeros(cropped_image.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (int(w * 0.05), int(h * 0.05), int(w * 0.9), int(h * 0.9))
        cv2.grabCut(cropped_image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        result_image = cropped_image * mask2[:, :, np.newaxis]
        return result_image, mask2
    except Exception as e:
        logging.error(f"배경 제거 중 오류 발생: {e}", exc_info=True)
        h, w = cropped_image.shape[:2]
        return cropped_image, np.zeros((h, w), np.uint8)


def preprocess_for_dark_text(image, pill_mask):
# 밝은 표면의 어두운 각인 (음각/그림자) 추출용

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 어두운 각인을 흰색으로 강조
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 19, 9  # 블록 크기 및 C값 조정
        )

        # 마스크를 적용하여 알약 영역만 남김
        masked_image = cv2.bitwise_and(thresh, thresh, mask=pill_mask)

        # Tesseract가 잘 읽도록 노이즈 제거 및 글씨 굵게
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(masked_image, cv2.MORPH_OPEN, kernel, iterations=1)

        # Tesseract는 (검은 글씨 / 흰 배경)을 선호하므로 반전시킴
        final_image = cv2.bitwise_not(cleaned)
        return final_image
    except Exception as e:
        logging.error(f"어두운 각인 전처리 중 오류: {e}", exc_info=True)
        return np.full_like(image, 255, dtype=np.uint8)  # 오류 시 흰색 이미지 반환


def preprocess_for_bright_text(image, pill_mask):
# 어두운 표면의 밝은 각인 (인쇄) 추출용
    
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 밝은 각인을 흰색으로 강조
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 19, 9  # 블록 크기 및 C값 조정
        )

        # 마스크를 적용하여 알약 영역만 남김
        masked_image = cv2.bitwise_and(thresh, thresh, mask=pill_mask)

        # Tesseract가 잘 읽도록 노이즈 제거 및 글씨 굵게
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(masked_image, cv2.MORPH_OPEN, kernel, iterations=1)

        # Tesseract는 (검은 글씨 / 흰 배경)을 선호하므로 반전시킴
        final_image = cv2.bitwise_not(cleaned)
        return final_image
    except Exception as e:
        logging.error(f"밝은 각인 전처리 중 오류: {e}", exc_info=True)
        return np.full_like(image, 255, dtype=np.uint8)  # 오류 시 흰색 이미지 반환


def preprocess_image(image_path):
# 이미지 경로를 입력받아 이미지를 로드
    try:
        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"이미지를 불러올 수 없습니다: {image_path}")
            return None
        return image
    except Exception as e:
        logging.error(f"이미지 로딩 중 오류 발생: {e}", exc_info=True)
        return None


# `preprocess_for_tesseract` 함수는 `imprint_analysis.py`에서
# 더 이상 호출하지 않으므로 삭제하거나 주석 처리해도 됩니다.
# 여기서는 하위 호환성을 위해 남겨두되, 내용은 비워둡니다.
def preprocess_for_tesseract(image, pill_mask):
    logging.warning("이 함수(preprocess_for_tesseract)는 사용되지 않아야 합니다. imprint_analysis.py를 확인하세요.")

    return np.full_like(image, 255, dtype=np.uint8)
