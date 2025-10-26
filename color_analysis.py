# color_analysis.py

import cv2
import numpy as np


# --- 다중 색상 알약 분석 ---
def analyze_pill_colors(pill_image_without_bg):
    """
    K-Means Clustering을 사용하여 알약의 주요 색상을 최대 2개까지 분석합니다.
    알약이 단일 색상인지, 두 가지 색상으로 조합되었는지 판별합니다.
    """
    # 이미지를 RGB로 변환하고 픽셀 데이터로 재구성
    image_rgb = cv2.cvtColor(pill_image_without_bg, cv2.COLOR_BGR2RGB)
    pixels = image_rgb.reshape(-1, 3)

    # 배경(검은색)을 제외한 실제 알약 픽셀만 필터링
    non_black_pixels = np.array([p for p in pixels if p.any()])
    if len(non_black_pixels) < 50:  # 분석에 필요한 최소 픽셀 수
        return None, ["색상 분석 불가"]

    # --- K-Means 실행 ---
    # K=4로 설정하여 알약의 양쪽 색상, 그림자, 하이라이트를 분리할 가능성을 높임
    samples = non_black_pixels.astype(np.float32)
    K = 4
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    attempts = 3
    flags = cv2.KMEANS_PP_CENTERS
    try:
        compactness, labels, centers = cv2.kmeans(samples, K, None, criteria, attempts, flags)
    except cv2.error:
        # 픽셀이 너무 적거나 색상 분리가 안될 경우 예외 처리
        return None, ["색상 분석 불가"]

    # 각 클러스터에 속한 픽셀 수 계산
    labels = labels.flatten()
    unique, counts = np.unique(labels, return_counts=True)

    # 픽셀 수가 적은 클러스터는 무시 (최소 5% 이상 차지해야 유의미한 색상으로 간주)
    min_pixel_percentage = 0.05
    significant_indices = [i for i, count in enumerate(counts) if count / len(samples) > min_pixel_percentage]

    if not significant_indices:
        # 모든 클러스터가 너무 작으면, 가장 큰 클러스터 하나만 사용
        dominant_rgb = centers[unique[np.argmax(counts)]].astype(int)
        dominant_color_name = map_rgb_to_color_name(dominant_rgb)
        return [dominant_rgb], [dominant_color_name]

    # 유의미한 클러스터들을 픽셀 수 기준으로 정렬
    sorted_significant_indices = sorted(significant_indices, key=lambda i: counts[i], reverse=True)

    # --- 상위 1~2개 색상 추출 및 분석 ---
    # 가장 많은 픽셀을 차지하는 색상
    top1_rgb = centers[unique[sorted_significant_indices[0]]].astype(int)
    top1_name = map_rgb_to_color_name(top1_rgb)

    final_rgb_colors = [top1_rgb]
    final_color_names = {top1_name}

    # 유의미한 클러스터가 2개 이상인 경우, 두 번째 색상 분석
    if len(sorted_significant_indices) > 1:
        top2_rgb = centers[unique[sorted_significant_indices[1]]].astype(int)
        top2_name = map_rgb_to_color_name(top2_rgb)

        # 두 색상의 이름이 다를 경우에만 두 번째 색상을 추가
        if top1_name != top2_name:
            final_rgb_colors.append(top2_rgb)
            final_color_names.add(top2_name)

    # 최종 결과를 정렬하여 반환 (예: ['빨강', '흰색'])
    return final_rgb_colors, sorted(list(final_color_names))


# --- HSV 색상 공간을 이용한 색상 매핑 (정확도 개선) ---
def map_rgb_to_color_name(rgb_color):
    """
    RGB 값을 HSV 색상 공간으로 변환하여 가장 가까운 색상 이름을 매핑합니다.
    """
    if rgb_color is None:
        return "알 수 없음"

    colors_rgb = {
        '하양': [255, 255, 255], '검정': [0, 0, 0], '회색': [128, 128, 128],
        '빨강': [255, 0, 0], '주황': [255, 165, 0], '노랑': [255, 255, 0],
        '초록': [0, 128, 0], '파랑': [0, 0, 255], '남색': [0, 0, 128],
        '보라': [128, 0, 128], '분홍': [255, 192, 203], '갈색': [165, 42, 42],
    }

    detected_color_np = np.uint8([[rgb_color]])
    detected_hsv = cv2.cvtColor(detected_color_np, cv2.COLOR_RGB2HSV)[0][0]

    if detected_hsv[1] < 40:
        if detected_hsv[2] > 200:
            return '흰색'
        elif detected_hsv[2] < 60:
            return '검정'
        else:
            return '회색'

    distances = []
    for name, value_rgb in colors_rgb.items():
        standard_color_np = np.uint8([[value_rgb]])
        standard_hsv = cv2.cvtColor(standard_color_np, cv2.COLOR_RGB2HSV)[0][0]
        hue_diff = abs(int(detected_hsv[0]) - int(standard_hsv[0]))
        hue_distance = min(hue_diff, 180 - hue_diff)
        distances.append((hue_distance, name))

    distances.sort(key=lambda x: x[0])
    return distances[0][1]