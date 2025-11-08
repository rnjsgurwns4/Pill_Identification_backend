import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, deltaE_cie76


# --- 색상 정의 (RGB 및 LAB) ---
color_dict_rgb = {
    '하양': [255, 255, 255],  '검정': [0, 0, 0],        '회색': [149, 165, 166],
    '빨강': [231, 76, 60],    '주황': [230, 126, 34],   '노랑': [241, 196, 15],
    '초록': [39, 174, 96],    '파랑': [52, 152, 219],   '남색': [0, 0, 128],
    '보라': [142, 68, 173],   '분홍': [231, 127, 153],  '갈색': [160, 82, 45]
}

# 모든 기준 색상의 LAB 값을 미리 계산하여 성능을 최적화합니다.
color_dict_lab = {
    name: rgb2lab(np.uint8([[rgb]]))
    for name, rgb in color_dict_rgb.items()
}


def map_rgb_to_color_name(rgb_color):
    """
    CIELAB 색 공간에서 Delta E 공식을 사용하여, 주어진 RGB 값과
    가장 시각적으로 가까운 색상 이름을 찾습니다.
    """
    if rgb_color is None: return "알 수 없음"
    input_lab = rgb2lab(np.uint8([[list(rgb_color)]]))
    min_dist = float('inf')
    closest_color = "알 수 없음"
    for name, standard_lab in color_dict_lab.items():
        dist = deltaE_cie76(input_lab, standard_lab)
        if dist < min_dist:
            min_dist = dist
            closest_color = name
    return closest_color


def analyze_pill_colors(pill_image_without_bg):
    """
    K-Means로 주요 색상 후보를 추출한 뒤, 후보 간의 시각적 유사도와
    반사/그림자 여부를 판단하여 단일/다중 색상을 최종 결정합니다.
    """
    try:
        image_rgb = cv2.cvtColor(pill_image_without_bg, cv2.COLOR_BGR2RGB)
        pixels = image_rgb.reshape(-1, 3)

        non_black_pixels = np.array([p for p in pixels if p.any()])
        n_clusters_to_use = 5
        if len(non_black_pixels) < 100:
            print(f"--- [Warn] 픽셀 수({len(non_black_pixels)}개)가 100개 미만입니다. K-Means 클러스터 수를 조절합니다.")
            if len(non_black_pixels) < 5:
                n_clusters_to_use = 1 # 5개 미만이면 클러스터를 1로 (평균색)
            else:
                n_clusters_to_use = 3 # 100개 미만 5개 이상이면 3으로 줄임
        

        kmeans = KMeans(n_clusters=n_clusters_to_use, n_init='auto', random_state=42)
        kmeans.fit(non_black_pixels)

        unique_labels, counts = np.unique(kmeans.labels_, return_counts=True)
        cluster_centers = kmeans.cluster_centers_.astype(int)

        min_pixel_percentage = 0.15
        significant_clusters = []
        for label, count in zip(unique_labels, counts):
            if count / len(non_black_pixels) > min_pixel_percentage:
                significant_clusters.append({'rgb': cluster_centers[label], 'count': count})

        if not significant_clusters:
            dominant_label = unique_labels[np.argmax(counts)]
            dominant_rgb = cluster_centers[dominant_label]
            color_name = map_rgb_to_color_name(dominant_rgb)

            return [dominant_rgb.tolist()], [color_name]

        sorted_clusters = sorted(significant_clusters, key=lambda x: x['count'], reverse=True)
        top1_rgb = sorted_clusters[0]['rgb']
        top1_name = map_rgb_to_color_name(top1_rgb)

        if len(sorted_clusters) == 1:

            return [top1_rgb.tolist()], [top1_name]

        top2_rgb = sorted_clusters[1]['rgb']
        top2_name = map_rgb_to_color_name(top2_rgb)

        # --- 단순하고 안정적인 최종 결정 로직 ---
        achromatic_colors = ['하양', '회색', '검정']
        top1_is_achromatic = top1_name in achromatic_colors
        top2_is_achromatic = top2_name in achromatic_colors

        # 1. 반사/그림자 처리: 한쪽이 유채색이고 다른 쪽이 무채색이면, 유채색을 정답으로 선택
        if top1_is_achromatic and not top2_is_achromatic:

            return [top2_rgb.tolist()], [top2_name]

        if not top1_is_achromatic and top2_is_achromatic:

            return [top1_rgb.tolist()], [top1_name]

        # 2. 색상 유사도 처리: 두 색상이 비슷하면(예: 주황과 갈색) 하나의 색으로 통일
        delta_e = float(deltaE_cie76(rgb2lab(np.uint8([[top1_rgb]])), rgb2lab(np.uint8([[top2_rgb]]))))
        SIMILARITY_THRESHOLD = 25.0
        if delta_e < SIMILARITY_THRESHOLD:

            return [top1_rgb.tolist()], [top1_name]

        # 3. 실제 다중 색상 처리: 위 두 경우에 해당하지 않으면, 실제로 두 가지 색을 가진 알약
        final_rgb_colors = [top1_rgb.tolist(), top2_rgb.tolist()]
        final_color_names = sorted([top1_name, top2_name])

        return final_rgb_colors, final_color_names

    except Exception as e:

        return [[0, 0, 0]], ["알 수 없음"]