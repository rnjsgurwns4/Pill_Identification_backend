
import cv2
import pytesseract
import logging
import re
import numpy as np
# image_preprocessing에서 두 개의 새로운 전문 함수를 가져옵니다.
from image_preprocessing import preprocess_for_dark_text, preprocess_for_bright_text




pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

def run_tesseract(image):
    """Tesseract OCR을 실행하고 결과를 정제하는 헬퍼 함수"""
    try:
        # Tesseract 설정: --psm 6은 이미지를 단일 텍스트 블록으로 간주
        config = r'--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        text = pytesseract.image_to_string(image, lang='eng', config=config)

        # OCR 결과 정제: 공백, 특수문자 제거
        cleaned_text = re.sub(r'[\W_]+', '', text).strip()
        return cleaned_text
    except Exception as e:
        return ""



# 각인 출력
def get_imprint(original_pill_image, pill_mask, debug=False):
    """
    알약 이미지에서 각인을 추출합니다.
    [수정] '어두운 각인'과 '밝은 각인' 두 가지 전처리를 모두 실행하고,
    최종적으로 문자열(string)만 반환합니다.
    """

    # --- 1. 어두운 각인 추출 시도 (밝은 표면용) ---
    preprocessed_dark = preprocess_for_dark_text(original_pill_image.copy(), pill_mask)
    text_from_dark = run_tesseract(preprocessed_dark)

    # --- 2. 밝은 각인 추출 시도 (어두운 표면용) ---
    preprocessed_bright = preprocess_for_bright_text(original_pill_image.copy(), pill_mask)
    text_from_bright = run_tesseract(preprocessed_bright)

    if debug:
        debug_image_dark = cv2.resize(preprocessed_dark, (300, 300))
        debug_image_bright = cv2.resize(preprocessed_bright, (300, 300))
        combined_debug_image = np.hstack((debug_image_dark, debug_image_bright))


    # --- 3. 결과 조합 ---
    combined_results = set()  # 중복 제거
    if text_from_dark:
        combined_results.add(text_from_dark)
    if text_from_bright:
        combined_results.add(text_from_bright)

    if not combined_results:
        # [수정] 튜플이 아닌 빈 문자열 반환
        return ""

    final_text = "/".join(combined_results)
    #confidence = 70.0  # 임시 신뢰도

    # [수정] 튜플이 아닌 최종 텍스트 문자열만 반환
    return final_text

