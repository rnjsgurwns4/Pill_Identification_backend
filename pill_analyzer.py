import cv2
import numpy as np
from PIL import ImageFont, Image, ImageDraw

# 로컬 모듈 임포트
from image_preprocessing import remove_background
from color_analysis import analyze_pill_colors
from shape_analysis import classify_shape_with_ai
from database_handler import find_best_match
from imprint_analysis import get_imprint as get_imprint_tesseract
from imprint_analysis_google import analyze_imprint_google 

from dotenv import load_dotenv

load_dotenv()
OCR_ENGINE = "google"
DEBUG_MODE = False

def draw_korean_text_on_image(image, text, position, pil_font):
# Pillow를 사용하여 이미지에 한글 텍스트를 그림
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
# 하나의 잘라낸 알약 이미지에 대해 전체 분석 파이프라인을 실행
    
    pill_without_bg, pill_mask = remove_background(cropped_pill_image.copy())
    all_shape_results = []
    all_color_sets = set()
    all_imprint_texts = []
    try:
        
        rgb_list, color_list = analyze_pill_colors(pill_without_bg)
        all_color_sets.update(color_list)
        
        gray_pill = cv2.cvtColor(pill_without_bg, cv2.COLOR_BGR2GRAY)
        _, binarized_image = cv2.threshold(gray_pill, 1, 255, cv2.THRESH_BINARY)
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
            
            contour_area = cv2.contourArea(pill_contour)
            min_rect = cv2.minAreaRect(pill_contour) 
            box_width, box_height = min_rect[1]
            box_area = box_width * box_height
            if box_area > 0:
                fill_ratio = contour_area / box_area
        else:
            fill_ratio = None
            
        if shape_model:
            shape_result = classify_shape_with_ai(smoothed_binarized_image, shape_model)
        
            if shape_result:
                primary_prediction = shape_result[0][0]
                if primary_prediction in ['타원형', '장방형'] and fill_ratio and fill_ratio > 0:
                    print(f"  --- AI: {primary_prediction}, Fill Ratio: {fill_ratio:.2f} ---")

                    scores_dict = dict(shape_result)
                    if fill_ratio < 0.85: # 85% 미만이면 타원형
                        if primary_prediction != '타원형':
                            if not(scores_dict['장방형'] > 0.9):
                                temp = scores_dict['타원형']
                                scores_dict['타원형'] = scores_dict['장방형']
                                scores_dict['장방형'] = temp
                    else: # 85% 이상이면 장방형
                        if primary_prediction != '장방형':
                            if not(scores_dict['타원형'] > 0.9):
                                temp = scores_dict['장방형']
                                scores_dict['장방형'] = scores_dict['타원형']
                                scores_dict['타원형'] = temp
                    shape_result_list = list(scores_dict.items())
                    shape_result_list.sort(key=lambda x: x[1], reverse=True)
                    formatted_list = [f"{name} ({conf:.2%})" for name, conf in shape_result_list]
                else:
                    formatted_list = [f"{name} ({conf:.2%})" for name, conf in shape_result]
                shape_result = ", ".join(formatted_list)
            all_shape_results.append(shape_result)

    except Exception as e:
        print(f"색상/모양 분석 중 심각한 에러 발생: {e}")
        return None


    print("각인 및 DB 조회 시작")
    imprint_text = ""
    if OCR_ENGINE == "google":
            
        imprint_text = analyze_imprint_google(cropped_pill_image.copy())

    elif OCR_ENGINE == "tesseract":
            
        imprint_text = get_imprint_tesseract(cropped_pill_image.copy(), pill_mask, debug=DEBUG_MODE)
    else:
            print(f"  - [오류] OCR_ENGINE 설정이 잘못되었습니다: {OCR_ENGINE}")

    print(f"  - 인식된 각인: '{imprint_text}'")
    if imprint_text:  # 빈 각인이 아니면 종합 리스트에 추가
        all_imprint_texts.append(imprint_text)
    combined_imprint = " ".join(sorted(list(set(all_imprint_texts))))
    combined_colors = " ".join(sorted(list(all_color_sets)))
    combined_shape_info = ""
    if all_shape_results:
        combined_shape_info = all_shape_results[0]
    print(combined_imprint, combined_colors, combined_shape_info)
        
    final_candidate_pills = find_best_match(pill_db, combined_shape_info, combined_colors, combined_imprint)
    
    return final_candidate_pills


def process_and_visualize_pills(original_image, pill_boxes, shape_model, pill_db, pil_font):
# 탐지된 모든 알약을 분석하고, 결과를 원본 이미지에 시각화
    
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

            image_with_results = draw_korean_text_on_image(image_with_results, label, (x1, y1), pil_font)
            cv2.rectangle(image_with_results, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            
            candidates_by_box.append(candidate_pills)
            
        else:
            # 결과 이미지에 그리기
            cv2.rectangle(image_with_results, (x1, y1), (x2, y2), (0, 0, 255), 2)
            image_with_results = draw_korean_text_on_image(image_with_results, label, (x1, y1), pil_font)
        pill_counter += 1
    

    return image_with_results, candidates_by_box

