
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import traceback
# --- AI 모양 분류 (모든 신뢰도 출력 및 비율 유지) ---
def classify_shape_with_ai(binarized_image, model, target_size=224):
    """
    학습된 AI 모델을 사용하여 이미지 모양을 분류하고, 모든 클래스의 신뢰도를 반환
    (가로세로 비율을 유지하고 패딩을 추가하여 왜곡 방지)
    """
    try:
        
        binarized_image = np.array(binarized_image, dtype=np.uint8)
        # 1. 원본 이미지의 가로, 세로 길이 확인
        h, w = binarized_image.shape
        
        # 2. 가로세로 비율을 유지하며 리사이징
        scale = target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_image = cv2.resize(binarized_image, (new_w, new_h))

        # 3. 검은색 정사각형 배경(패드) 생성
        pad = np.zeros((target_size, target_size), dtype=np.uint8)
        
        # 4. 리사이징된 이미지를 배경 중앙에 배치
        top_left_x = (target_size - new_w) // 2
        top_left_y = (target_size - new_h) // 2
        pad[top_left_y:top_left_y + new_h, top_left_x:top_left_x + new_w] = resized_image
        
        # 5. 모델 입력에 맞게 3채널(RGB)로 변환 및 전처리
        input_image_rgb = cv2.cvtColor(pad, cv2.COLOR_GRAY2RGB)
        input_array = img_to_array(input_image_rgb)
        scaled_array = input_array / 255.0
        

        # 6. 예측 실행
        predictions = model.predict(scaled_array[np.newaxis, ...])[0]
        
        shape_map = {0: '원형', 1: '육각형', 2:'타원형'} # 모델 학습 시 클래스 순서와 동일해야 함
        
        # 모든 클래스에 대한 신뢰도 문자열 생성
        results = []
        for i, confidence in enumerate(predictions):
            shape_name = shape_map.get(i, f"클래스 {i}")
            results.append((shape_name, confidence))
        
        # 신뢰도가 높은 순으로 정렬
        results.sort(key=lambda x: x[1], reverse=True)
        
        # 문자열로 변환하여 반환
        return ", ".join([f"{name} ({conf:.2%})" for name, conf in results])
        
    except Exception as e:
        print(f"    - 모양 분류 모델 로딩 또는 예측 실패: {e}")
        traceback.print_exc() 
        return "AI 모델 분석 실패 (임시)"
