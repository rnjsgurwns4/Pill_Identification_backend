

import cv2
import numpy as np


# --- GrabCut 알고리즘 기반 배경 제거 ---
def remove_background(pill_image):
    """
    GrabCut 알고리즘을 사용하여 내부 질감이 복잡하거나 배경과 색이 비슷한
    이미지에서도 알약을 정교하게 분리
    """
    # 이미지가 존재하지 않을 경우, 빈 검은 화면 반환
    if pill_image is None or pill_image.size == 0:
        blank_mask = np.zeros((100, 100), dtype="uint8")
        blank_image = np.zeros((100, 100, 3), dtype="uint8")
        return blank_image, blank_mask
    
    
    # 마스크: 객체와 배경을 구분하는 정보를 저장할 행렬. 0으로 초기화
    # bgdModel, fgdModel: 배경과 전경의 색상 분포 모델을 저장할 임시 배열
    mask = np.zeros(pill_image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    # 이미지의 가로나 세로가 너무 작으면 Otsu 이진화를 사용
    h, w = pill_image.shape[:2]
    if h < 20 or w < 20:
        gray = cv2.cvtColor(pill_image, cv2.COLOR_BGR2GRAY) #흑백화
        _, final_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result = cv2.bitwise_and(pill_image, pill_image, mask=final_mask)
        return result, final_mask

    # 알약이 이미지에 꽉 차 있을 것을 대비해, 마진을 1픽셀로 최소화
    rect = (1, 1, w - 2, h - 2)

    # GrabCut 알고리즘 실행
    try:
        cv2.grabCut(pill_image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    # 실패 시 Otsu 이진화
    except Exception as e:
        print(f"    - GrabCut 실패: {e}. Otsu 방식으로 전환합니다.")
        gray = cv2.cvtColor(pill_image, cv2.COLOR_BGR2GRAY)
        _, final_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result = cv2.bitwise_and(pill_image, pill_image, mask=final_mask)
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
    
    
    result = pill_image * final_mask[:, :, np.newaxis]
    
    final_mask_255 = final_mask * 255
    
    return result, final_mask_255


# 안 씀(깃허브에서 삭제하기)
# --- ★ 수정된 부분: 그레이스케일 기반 각인 전처리 함수에 노이즈 제거 추가 ---
def preprocess_for_imprint(original_pill_image, pill_mask):
    """
    원본 알약 이미지와 마스크를 사용하여 각인을 강조하는 전처리를 수행합니다.
    """
    gray = cv2.cvtColor(original_pill_image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    denoised = cv2.GaussianBlur(equalized, (3, 3), 0)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    blackhat = cv2.morphologyEx(denoised, cv2.MORPH_BLACKHAT, kernel)
    """
    # 이진화(Thresholding)를 통해 희미한 글자를 선명한 흰색으로 만듭니다.
    _, thresholded = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # ★★★ 추가된 부분: 형태학적 열림(Opening) 연산으로 노이즈 제거 ★★★
    # 작은 커널을 사용하여 이미지에서 작은 점 같은 노이즈를 제거합니다.
    opening_kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, opening_kernel, iterations=1)
    """
    
    # 최종적으로 마스크를 적용하여 배경(알약이 아닌 부분)을 제거
    imprint_only = cv2.bitwise_and(blackhat, blackhat, mask=pill_mask)
    
    return imprint_only



