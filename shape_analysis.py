
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import traceback


def classify_shape_with_ai(binarized_image, model, target_size=224):
# 학습된 AI 모델을 사용하여 이미지 모양을 분류하고, 모든 클래스의 신뢰도를 반환

    try:
        binarized_image = np.array(binarized_image, dtype=np.uint8)
        # 원본 이미지의 가로, 세로 길이 확인
        h, w = binarized_image.shape
        
        # 가로세로 비율을 유지하며 리사이징
        scale = target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_image = cv2.resize(binarized_image, (new_w, new_h))

        # 검은색 정사각형 배경(패드) 생성
        pad = np.zeros((target_size, target_size), dtype=np.uint8)
        
        # 리사이징된 이미지를 배경 중앙에 배치
        top_left_x = (target_size - new_w) // 2
        top_left_y = (target_size - new_h) // 2
        pad[top_left_y:top_left_y + new_h, top_left_x:top_left_x + new_w] = resized_image
        
        # 모델 입력에 맞게 3채널(RGB)로 변환 및 전처리
        input_image_rgb = cv2.cvtColor(pad, cv2.COLOR_GRAY2RGB)
        input_array = img_to_array(input_image_rgb)
        scaled_array = input_array / 255.0
        
        final_input = scaled_array[np.newaxis, ...]

        # 예측 실행
        predictions = model.predict(scaled_array[np.newaxis, ...])[0]
        print("model complete") 
        shape_map = {0: '원형', 1: '타원형', 2:'장방형'}
        
        results_list = []
        for i, confidence in enumerate(predictions):
            shape_name = shape_map.get(i, f"unknown_{i}")
            results_list.append((shape_name, float(confidence))) # 튜플로 저장
        
        results_list.sort(key=lambda x: x[1], reverse=True)

        return results_list
        
    except Exception as e:
        print(f"    - 모양 분류 모델 로딩 또는 예측 실패: {e}")
        traceback.print_exc() 
        return "AI 모델 분석 실패 (임시)"

