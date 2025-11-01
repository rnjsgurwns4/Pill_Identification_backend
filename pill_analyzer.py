

import cv2
import numpy as np
from PIL import ImageFont, Image, ImageDraw

# 로컬 모듈 임포트
from image_preprocessing import remove_background
from color_analysis import get_dominant_color
from shape_analysis import classify_shape_with_ai
from database_handler import find_best_match
from imprint_analysis import get_imprint

# --- 유틸리티 함수를 app.py에서 여기로 이동 ---
def draw_korean_text_on_image(image, text, position, pil_font):
    """ Pillow를 사용하여 이미지에 한글 텍스트를 그림 """
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    try: _, _, text_width, text_height = pil_font.getbbox(text)
    except AttributeError: text_width, text_height = pil_font.getsize(text)

    x, y = position
    # 텍스트 배경 사각형 그리기
    draw.rectangle(((x, y - text_height - 10), (x + text_width + 10, y)), fill=(0, 255, 0))
    # 텍스트 그리기
    draw.text((x + 5, y - text_height - 7), text, font=pil_font, fill=(0, 0, 0))
    
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def analyze_single_pill(cropped_pill_image, shape_model, pill_db):
    """ 하나의 잘라낸 알약 이미지에 대해 전체 분석 파이프라인을 실행 """
    
    pill_without_bg, pill_mask = remove_background(cropped_pill_image.copy())

    try:
        
        _, color_candidates = get_dominant_color(pill_without_bg)
        
        gray_pill = cv2.cvtColor(pill_without_bg, cv2.COLOR_BGR2GRAY)
        _, binarized_image = cv2.threshold(gray_pill, 10, 255, cv2.THRESH_BINARY)
        smoothed_binarized_image = binarized_image.copy()

        
        contours, _ = cv2.findContours(binarized_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        
        

        shape_result = None
        if contours:
            pill_contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(pill_contour, True)
            epsilon = 0.005 * perimeter
            approximated_contour = cv2.approxPolyDP(pill_contour, epsilon, True)
            smoothed_binarized_image = np.zeros_like(binarized_image)
            cv2.drawContours(smoothed_binarized_image, [approximated_contour], -1, (255), -1)
        
        if shape_model:
            
            shape_result = classify_shape_with_ai(smoothed_binarized_image, shape_model)
            
    
       

    except Exception as e:
        # 혹시 모를 예외(에러)를 잡기 위해 try-except 구문 추가
        print(f"!!!!!! 색상/모양 분석 중 심각한 에러 발생: {e}")
        # 에러 발생 시 None을 반환하거나 적절한 처리가 필요할 수 있습니다.
        return None


    print("--- [4/5] 각인 및 DB 조회 시작 ---")
    imprint_text = get_imprint(cropped_pill_image.copy(), pill_mask)
    
    candidate_pills = find_best_match(pill_db, shape_result, color_candidates, imprint_text)
    
    return candidate_pills

# --- app.py의 for 루프 로직을 담당할 새로운 메인 함수 ---
def process_and_visualize_pills(original_image, pill_boxes, shape_model, pill_db, pil_font):
    """
    탐지된 모든 알약을 분석하고, 결과를 원본 이미지에 시각화
    
    Returns:
        tuple: (결과가 그려진 이미지, 후보 알약 데이터 리스트)
    """
    
    candidates_by_box = []
    # 원본 이미지를 복사하여 여기에 그림
    image_with_results = original_image.copy()
    
    pill_counter = 1
    

    for box in pill_boxes:
        label = f"알약{pill_counter}"
        x1, y1, x2, y2 = box
        cropped_pill = original_image[y1:y2, x1:x2]
        
        # 각 알약 분석
        candidate_pills = analyze_single_pill(cropped_pill, shape_model, pill_db)
        
        # 분석 결과를 이미지에 그리고 응답 데이터 구성
        if candidate_pills:
            top_candidate = candidate_pills[0]
            #label = f"{top_candidate['pill_info']}"
            # 결과 이미지에 그리기
            image_with_results = draw_korean_text_on_image(image_with_results, label, (x1, y1), pil_font)
            cv2.rectangle(image_with_results, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            
            candidates_by_box.append(candidate_pills)
            
        else:
            # 결과 이미지에 그리기
            cv2.rectangle(image_with_results, (x1, y1), (x2, y2), (0, 0, 255), 2)
            image_with_results = draw_korean_text_on_image(image_with_results, label, (x1, y1), pil_font)
        pill_counter += 1
    
    return image_with_results, candidates_by_box