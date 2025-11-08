import os
import re
import cv2
import logging
from google.cloud import vision






# ----------------------------------------------------------------------

def analyze_imprint_google(original_pill_image):
    
    
    KEY_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    try:
        if KEY_PATH is None:
            logging.error("=" * 50)
            logging.error(" [오류] .env 파일에 'GOOGLE_APPLICATION_CREDENTIALS'가 설정되지 않았습니다.")
            logging.error("=" * 50)
            client = None
        elif not os.path.exists(KEY_PATH):
            logging.error("=" * 50)
            logging.error(f" [오류] .env에 설정된 .json 파일 경로를 찾을 수 없습니다: {KEY_PATH}")
            logging.error("=" * 50)
            client = None
        else:
            client = vision.ImageAnnotatorClient()
            logging.info("--- Google Vision 클라이언트 초기화 성공! ---")

    except Exception as e:
        logging.error(f"Google Vision 클라이언트 초기화 오류: {e}")
        logging.error("1. .env의 .json 파일이 올바른지, 2. Google Cloud에서 'Cloud Vision API'를 '사용 설정'했는지 확인하세요.")
        client = None
    
    """
    YOLO가 잘라낸 '원본' 이미지를 Google Vision API로 분석하여 각인 텍스트를 추출
    """
    if client is None:
        logging.error("Google Vision 클라이언트가 초기화되지 않아 OCR을 건너뜁니다.")
        return ""

    print("    - Google Vision API 분석 시도...")

    try:
        # OpenCV 이미지(numpy)를 Google API가 읽을 수 있는 bytes로 변환
        _, buffer = cv2.imencode('.jpg', original_pill_image)
        image_bytes = buffer.tobytes()

        image = vision.Image(content=image_bytes)

        # API 호출 (텍스트 감지)
        response = client.text_detection(image=image)
        texts = response.text_annotations

        if response.error.message:
            print(f"    - Google API 오류: {response.error.message}")
            return ""

        # 결과 파싱
        if texts:
            # 첫 번째 결과(texts[0])는 이미지의 모든 텍스트를 합친 것입니다.
            full_text = texts[0].description
            # 영숫자만 남기고 대문자화
            cleaned_text = re.sub(r'[\W_]+', '', full_text).strip().upper()
            print(f"      => 결과: '{cleaned_text}'")
            return cleaned_text
        else:
            print("      => 결과: ''")
            return ""

    except Exception as e:
        print(f"    - Google Vision API 호출 오류: {e}")
        return ""