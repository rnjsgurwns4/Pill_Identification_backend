# database_handler.py

import pandas as pd
import numpy as np

# 색상명과 RGB 값 매핑 (수정/추가 가능)
# 데이터베이스에 있는 다른 색상(예: '투명', '자홍')이 있다면 여기에 추가해주세요.
COLOR_RGB_MAP = {
    '하양': [255, 255, 255], '검정': [0, 0, 0], '회색': [128, 128, 128],
    '빨강': [255, 0, 0], '주황': [255, 165, 0], '노랑': [255, 255, 0],
    '초록': [0, 128, 0], '파랑': [0, 0, 255], '남색': [0, 0, 128],
    '보라': [128, 0, 128], '분홍': [255, 192, 203], '갈색': [165, 42, 42],
}

# RGB 색 공간에서 이론상 가장 먼 거리
MAX_COLOR_DIST = np.sqrt(255 ** 2 * 3)


def get_color_distance(color_name1, color_name2):
    """ 두 색상 이름 간의 RGB 공간에서의 유클리드 거리를 계산합니다. """
    rgb1 = COLOR_RGB_MAP.get(color_name1)
    rgb2 = COLOR_RGB_MAP.get(color_name2)

    if rgb1 is None or rgb2 is None:
        # 맵에 없는 색상이면 최대 거리를 반환하여 유사도를 0으로 만듭니다.
        return MAX_COLOR_DIST

    return np.linalg.norm(np.array(rgb1) - np.array(rgb2))


def calculate_color_similarity_score(identified_colors_str, db_colors_str):
    """
    인식된 색상과 DB의 색상 간의 유사도 점수를 계산합니다.
    점수는 0 (완전 다름) 에서 1 (완전 같음) 사이의 값입니다.
    """
    identified_colors = set(identified_colors_str.split())
    db_colors = set(db_colors_str.split())

    if not identified_colors or not db_colors:
        return 0

    total_max_similarity = 0

    # 각 인식된 색상에 대해 DB 색상 중 가장 유사한 것을 찾습니다.
    for id_color in identified_colors:
        max_similarity_for_color = 0
        for db_color in db_colors:
            if id_color == db_color:
                similarity = 1.0
            else:
                distance = get_color_distance(id_color, db_color)
                similarity = max(0, 1 - (distance / MAX_COLOR_DIST))

            if similarity > max_similarity_for_color:
                max_similarity_for_color = similarity

        total_max_similarity += max_similarity_for_color

    # 평균 최대 유사도를 최종 점수로 반환합니다.
    return total_max_similarity / len(identified_colors)


def load_database(db_path):
    """
    CSV 데이터베이스를 로드하고, 모든 데이터를 문자열로 변환하여 반환
    """
    try:
        df = pd.read_csv(db_path, encoding='cp949')
        # 모든 열의 데이터를 문자열 타입으로 변환하여 타입 오류 방지
        for col in df.columns:
            df[col] = df[col].astype(str)
        print(f"'{db_path}' 데이터베이스를 성공적으로 불러왔습니다.")
        return df.to_dict('records')
    except Exception as e:
        print(f"데이터베이스 로딩 오류: {e}")
        return None


def get_char_type(s):
    """문자열의 종류(알파벳, 숫자, 혼합 등)를 반환하는 헬퍼 함수"""
    if not s:  # 문자열이 비어있는 경우
        return "none"
    if s.isalpha():
        return "alpha"
    if s.isnumeric():
        return "numeric"
    if s.isalnum():
        return "alnum"
    return "other"


def calculate_score(row, shape_probabilities, colors, imprint):
    """
    데이터베이스의 약 정보와 분석된 정보를 비교하여 유사도 점수를 계산
    (점수가 높을수록 더 유사함)
    """
    score = 0
    MAX_SHAPE_SCORE = 30
    MAX_COLOR_SCORE = 30
    MAX_IMPRINT_SCORE = 40

    # 1. 모양 점수: AI의 예측 확률에 따라 가중치 부여
    shape_score = 0
    db_shape = row['shape']
    if db_shape in shape_probabilities:
        # 확률값(%)을 100으로 나누어 0~1 사이의 값으로 변환
        probability = shape_probabilities[db_shape] / 100.0
        shape_score = MAX_SHAPE_SCORE * probability
    score += shape_score

    # 2. 색상 점수: 색상 유사도에 따라 점수 부여 (0~30점)
    color_similarity = calculate_color_similarity_score(colors, row['color'])
    score += color_similarity * MAX_COLOR_SCORE

    # 3. 각인 점수 계산
    imprint_recognized = imprint.upper()

    if not imprint_recognized:
        imprint_score = 0
    else:
        imprint1_db = str(row.get('text', '')).upper()
        imprint2_db = str(row.get('text2', '')).upper()

        dist1 = levenshtein_distance(imprint_recognized, imprint1_db)
        dist2 = levenshtein_distance(imprint_recognized, imprint2_db)

        min_dist = min(dist1, dist2)

        db_imprint_to_compare = imprint1_db if dist1 <= dist2 else imprint2_db
        recognized_type = get_char_type(imprint_recognized)
        db_type = get_char_type(db_imprint_to_compare)

        if recognized_type in ["alpha", "numeric"] and db_type in ["alpha", "numeric"] and recognized_type != db_type:
            imprint_score = 0
        else:
            imprint_score = max(0, MAX_IMPRINT_SCORE - (min_dist * 10))

    score += imprint_score
    return score


def levenshtein_distance(s1, s2):
    """
    두 문자열 간의 레벤슈타인 거리를 계산
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def find_best_match(pill_db, identified_shape_info, identified_colors, identified_imprint):
    """
    분석된 정보를 바탕으로 데이터베이스에서 가장 일치하는 알약 후보를 찾음.
    """

    # [수정됨] 문자열, 딕셔너리 등 어떤 형태로 들어와도 처리 가능하도록 파싱 로직 추가
    shape_probabilities = {}
    if isinstance(identified_shape_info, str) and identified_shape_info:
        # '타원형 (78.55%), 원형 (17.31%)' 과 같은 문자열을 파싱하여 딕셔너리로 변환
        parts = identified_shape_info.split(', ')
        for part in parts:
            try:
                shape, prob_str = part.split(' (')
                prob = float(prob_str.replace('%)', ''))
                shape_probabilities[shape] = prob
            except (ValueError, IndexError):
                continue  # 파싱에 실패하는 경우(예: '(A)...' 등)는 무시
    elif isinstance(identified_shape_info, dict):
        shape_probabilities = identified_shape_info

    primary_candidates = []
    for row in pill_db:
        # shape_probabilities의 키(모양 이름)를 기준으로 후보군 필터링
        shape_match = row['shape'] in shape_probabilities
        color_match = any(color in row['color'] for color in identified_colors.split())

        if shape_match or color_match:
            primary_candidates.append(row)

    if not primary_candidates:
        print("  - [검색 실패] 데이터베이스에서 어떤 후보도 찾을 수 없습니다.")
        return []

    candidates = []
    for row in primary_candidates:
        # 파싱된 shape_probabilities 딕셔너리를 점수 계산에 사용
        score = calculate_score(row, shape_probabilities, identified_colors, identified_imprint)
       # imprint_display = f"앞:{row.get('text', '')}/뒤:{row.get('text2', '')}"
        #pill_info = f"{row['name']} ({row['shape']}, {row['color']}, {imprint_display})"
        #candidates.append({'pill_info': pill_info, 'score': score})
        pill_name = row.get('name', '알 수 없음')
        pill_code = row.get('code', 'N/A') # CSV에서 'code' 열을 가져옵니다.
        
        candidates.append({
            'pill_info': pill_name,  # 목표 1: 'pill_info'에 이름만 저장
            'code': pill_code,      # 목표 1: 'code' 정보 추가
            'score': score
        })

    candidates.sort(key=lambda x: x['score'], reverse=True)
    return [c for c in candidates if c['score'] > 0][:10]


def calculate_search_score(row, search_name, search_shape, search_color, search_imprint):
    """
    텍스트 검색어를 기반으로 DB의 row와 점수를 계산합니다.
    (점수가 높을수록 더 유사함)
    """
    score = 0
    # 점수 가중치 (총 100점)
    MAX_NAME_SCORE = 40
    MAX_IMPRINT_SCORE = 30
    MAX_SHAPE_SCORE = 15
    MAX_COLOR_SCORE = 15

    # 1. 이름 점수 (레벤슈타인 거리 사용)
    db_name = str(row.get('name', '')).upper()
    search_name = search_name.upper()
    if search_name and db_name:
        dist = levenshtein_distance(search_name, db_name)
        # 이름 길이에 따라 거리를 정규화하여 유사도 계산
        similarity = max(0, 1 - (dist / max(len(search_name), len(db_name))))
        score += similarity * MAX_NAME_SCORE

    # 2. 각인 점수 (기존 로직 활용)
    search_imprint = search_imprint.upper()
    if search_imprint:
        imprint1_db = str(row.get('text', '')).upper()
        imprint2_db = str(row.get('text2', '')).upper()
        
        dist1 = levenshtein_distance(search_imprint, imprint1_db)
        dist2 = levenshtein_distance(search_imprint, imprint2_db)
        min_dist = min(dist1, dist2)
        
        # 각인 길이에 따라 거리를 정규화하여 유사도 계산
        similarity = max(0, 1 - (min_dist / max(1, len(search_imprint))))
        score += similarity * MAX_IMPRINT_SCORE

    # 3. 모양 점수 (유사도)
    db_shape = str(row.get('shape', '')).upper()
    search_shape = search_shape.upper()
    if search_shape and db_shape:
        if search_shape == db_shape: # 모양은 정확히 일치할 때만 점수
            score += MAX_SHAPE_SCORE
        # (만약 '타'만 쳐도 '타원형'이 검색되게 하려면 'if search_shape in db_shape:' 사용)

    # 4. 색상 점수 (기존 로직 활용)
    db_color = str(row.get('color', ''))
    search_color = search_color.strip()
    if search_color and db_color:
        # '하양 분홍' -> '하양' 검색도 가능하도록 기존 함수 사용
        color_similarity = calculate_color_similarity_score(search_color, db_color)
        score += color_similarity * MAX_COLOR_SCORE

    return score


# --- [신규 추가] 텍스트 검색 메인 함수 ---

def find_match_by_text(pill_db, name, shape, color, imprint):
    """
    텍스트 검색어를 받아 DB에서 가장 유사한 알약 리스트를 반환
    """
    candidates = []
    
    # DB의 모든 알약을 순회하며 점수 계산
    for row in pill_db:
        score = calculate_search_score(row, 
                                     search_name=name, 
                                     search_shape=shape, 
                                     search_color=color, 
                                     search_imprint=imprint)
        
        # 점수가 0보다 큰 경우에만 후보에 추가
        if score > 0:
            candidates.append({
                'pill_info': row.get('name', '알 수 없음'),
                'code': row.get('code', 'N/A'),
                'score': round(score, 2)
            })

    # 점수가 높은 순으로 정렬
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    # 상위 10개만 반환
    return candidates[:10]