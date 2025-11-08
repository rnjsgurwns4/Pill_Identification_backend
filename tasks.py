# tasks.py

import cv2
import base64
import numpy as np
from celery import Celery
from tensorflow.keras.models import load_model
from PIL import ImageFont

# 로컬 모듈 임포트
from database_handler import load_database
from object_detection import detect_pills
from pill_analyzer import process_and_visualize_pills

# --- Celery 설정 ---
# 'redis://localhost:6379/0'는 Redis 서버의 주소
# broker: 작업(task)을 저장
# backend: 작업 결과를 저장
celery_app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

celery_app.conf.update(
    # 결과가 Redis 백엔드에 저장된 후 3600초 (1시간)가 지나면 자동으로 삭제
    result_expires=3600,
)

# --- 모델, DB, 폰트 로드 ---
SHAPE_MODEL_PATH = "weights/shape_model.h5"
DB_PATH = "database/pill.csv"
YOLO_MODEL_PATH = "weights/detection_model.pt"
FONT_PATH_BOLD = "fonts/malgunbd.ttf"
FONT_SIZE = 18

OCR_ENGINE = "google"

DEBUG_MODE = False

PILL_DB = load_database(DB_PATH)
SHAPE_MODEL = None
PIL_FONT = ImageFont.truetype(FONT_PATH_BOLD, FONT_SIZE)


# --- Celery Task 생성 ---
@celery_app.task
def analyze_pill_image_task(image_string):
    """
    Base64 인코딩된 이미지 문자열을 받아 알약 분석을 수행하는 Celery Task.
    """
    global SHAPE_MODEL

    # 2. 이 워커 프로세스에서 모델이 아직 로드되지 않았다면 로드
    if SHAPE_MODEL is None:
        SHAPE_MODEL = load_model(SHAPE_MODEL_PATH)

    # Base64 문자열을 다시 이미지(numpy array)로 디코딩
    npimg = np.frombuffer(base64.b64decode(image_string), np.uint8)
    original_image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # 알약 탐지
    pill_boxes = detect_pills(original_image, YOLO_MODEL_PATH)



    # 분석 및 시각화 (시간이 오래 걸리는 부분)
    processed_image, candidates = process_and_visualize_pills(
        original_image, pill_boxes, SHAPE_MODEL, PILL_DB, PIL_FONT
    )





    # 결과 이미지를 다시 Base64 문자열로 인코딩
    _, buffer = cv2.imencode('.jpg', processed_image)
    img_str = base64.b64encode(buffer).decode('utf-8')

    final_pill_results = []
        
    # candidates_by_box는 [ [알약1후보A, 알약1후보B], [알약2후보A] ] 형태의 2차원 리스트
    for pill_candidates_for_one_box in candidates:
            
        # 이 상자(box)의 모든 후보 결과를 담을 리스트
        current_box_results = [] 
            
        if not pill_candidates_for_one_box:
            final_pill_results.append(current_box_results) # 빈 리스트라도 추가
            continue

        # 이 상자(box)의 모든 후보를 순회
        for candidate in pill_candidates_for_one_box:
            item_code = candidate.get('code')
            pill_name = candidate.get('pill_info')
            image_url = candidate.get('image_url', '')
                    
            
            # 현재 상자의 결과 리스트(current_box_results)에 추가
            current_box_results.append({ 
                'pill_info': pill_name,
                'code': item_code,
                'image': image_url
            })
            
        # 최종 결과 리스트(final_pill_results)에 "현재 상자의 결과 리스트"를 통째로 추가
        final_pill_results.append(current_box_results) 

    # 최종 결과 반환
    return {
        'processed_image': img_str,
        'pill_results': final_pill_results
    }

