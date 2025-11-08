import cv2
import numpy as np
import logging


def remove_background(cropped_image):
    """
    GrabCut 알고리즘을 사용하여 이미 잘라낸 알약 이미지에서 배경을 제거합니다.
    """
    """
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
    """
    mask = np.zeros(cropped_image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    # 이미지의 가로나 세로가 너무 작으면 Otsu 이진화를 사용
    h, w = cropped_image.shape[:2]
    if h < 20 or w < 20:
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY) #흑백화
        _, final_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result = cv2.bitwise_and(cropped_image, cropped_image, mask=final_mask)
        return result, final_mask

    # 알약이 이미지에 꽉 차 있을 것을 대비해, 마진을 1픽셀로 최소화
    rect = (1, 1, w - 2, h - 2)

    # GrabCut 알고리즘 실행
    try:
        cv2.grabCut(cropped_image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    # 실패 시 Otsu 이진화
    except Exception as e:
        print(f"    - GrabCut 실패: {e}. Otsu 방식으로 전환합니다.")
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        _, final_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result = cv2.bitwise_and(cropped_image, cropped_image, mask=final_mask)
        return result, final_mask

    # 0: 확실한 배경, 2: 아마도 배경 -> 0 (배경)으로 변경
    # 1: 확실한 전경, 3: 아마도 전경 -> 1 (전경)으로 변경
    # np.where 함수를 이용해 최종적으로 배경은 0, 전경(알약)은 1인 바이너리 마스크를 생성
    final_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    
    # 노이즈 제거 (Morphology 연산)
    # 마스크에 생길 수 있는 작은 구멍(노이즈)을 메우기
    # MORPH_ELLIPSE: 타원 모양의 커널(필터)을 생성
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    # MORPH_CLOSE: 팽창(Dilate) 후 침식(Erode)을 수행하여 객체 내부의 작은 구멍을 채우기
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    
    result = cropped_image * final_mask[:, :, np.newaxis]
    
    final_mask_255 = final_mask * 255
    
    return result, final_mask_255
    

def preprocess_for_dark_text(image, pill_mask):
    """
    [전문 함수 1] 밝은 표면의 '어두운' 각인 (음각/그림자) 추출용.
    어두운 부분을 찾아 흰색으로(THRESH_BINARY_INV) 변환합니다.
    """
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
    """
    [전문 함수 2] 어두운 표면의 '밝은' 각인 (인쇄) 추출용.
    밝은 부분을 찾아 흰색으로(THRESH_BINARY) 변환합니다.
    """
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
    """
    이미지 경로를 입력받아 이미지를 로드합니다.
    """
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