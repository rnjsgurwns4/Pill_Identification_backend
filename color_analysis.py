

import cv2
import numpy as np
from sklearn.cluster import KMeans

# --- K-Means 색상 분석 ---
def get_dominant_color(pill_image_without_bg):
    """
    K-Means Clustering을 사용하여 배경이 제거된 이미지에서 주요 색상 찾기
    """
    # 이미지를 RGB로 변환하고 픽셀 데이터로 재구성
    image_rgb = cv2.cvtColor(pill_image_without_bg, cv2.COLOR_BGR2RGB)
    pixels = image_rgb.reshape(-1, 3)
    
    # 배경(검은색)을 제외한 실제 알약 픽셀만 필터링
    non_black_pixels = np.array([p for p in pixels if p.any()])
    if len(non_black_pixels) < 5:
        return None, "색상 분석 불가"

    # K-Means를 사용하여 3개의 주요 색상 클러스터를 찾음
    kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto').fit(non_black_pixels)
    
    # 가장 많은 픽셀을 차지하는 클러스터의 중심 색상을 주요 색상으로 선택
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_rgb = kmeans.cluster_centers_[unique[np.argmax(counts)]].astype(int)
    
    return dominant_rgb, map_rgb_to_color_name(dominant_rgb)

# --- HSV 색상 공간을 이용한 색상 매핑 ---
def map_rgb_to_color_name(rgb_color, top_n=3):
    """
    RGB 값을 HSV 색상 공간으로 변환하여 가장 가까운 색상 이름을 매핑
    (인간의 색상 인지 방식과 더 유사하여 정확도가 높음)
    """
    if rgb_color is None:
        return "알 수 없음"

    # 비교를 위한 표준 색상 정의 (RGB)
    colors_rgb = {
        '하양': [255, 255, 255], '검정': [0, 0, 0], '회색': [128, 128, 128],
        '빨강': [255, 0, 0], '주황': [255, 165, 0], '노랑': [255, 255, 0],
        '초록': [0, 128, 0], '파랑': [0, 0, 255], '남색': [0, 0, 128],
        '보라': [128, 0, 128], '분홍': [255, 192, 203], '갈색': [165, 42, 42],
    }

    # 1. 감지된 RGB 색상을 HSV로 변환
    # OpenCV는 3차원 배열을 요구
    detected_color_np = np.uint8([[rgb_color]])
    detected_hsv = cv2.cvtColor(detected_color_np, cv2.COLOR_RGB2HSV)[0][0]
    
    # 2. 흰색/검정색 특별 처리 (채도와 명도 기준)
    # 채도(S)가 낮으면 무채색, 명도(V)가 높으면 흰색, 낮으면 검정색
    if detected_hsv[1] < 35: # 채도가 매우 낮으면
        if detected_hsv[2] > 200: # 명도가 매우 높으면
            return '흰색'
        elif detected_hsv[2] < 50: # 명도가 매우 낮으면
            return '검정'
    
    distances = []
    # 모든 표준 유채색과의 Hue 거리 계산
    for name, value_rgb in colors_rgb.items():
        standard_color_np = np.uint8([[value_rgb]])
        standard_hsv = cv2.cvtColor(standard_color_np, cv2.COLOR_RGB2HSV)[0][0]

        hue_diff = abs(int(detected_hsv[0]) - int(standard_hsv[0]))
        hue_distance = min(hue_diff, 180 - hue_diff)
        distances.append((hue_distance, name))
            
    # 거리가 가까운 순으로 정렬
    distances.sort(key=lambda x: x[0])
    
    # 상위 N개의 색상 이름만 추출하여 반환
    return [name for dist, name in distances[:top_n]]