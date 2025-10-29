# app.py

import base64
from flask import Flask, request, jsonify
from celery.result import AsyncResult

# Celery Task와 외부 API 핸들러를 임포트합니다.
from tasks import analyze_pill_image_task
from api_handler import get_pill_details_from_api
from database_handler import load_database, find_match_by_text
# Flask 앱을 초기화합니다.
app = Flask(__name__)

DB_PATH = 'database/pill.csv'
pill_db = load_database(DB_PATH)
if pill_db is None:
    print(f"경고: '{DB_PATH}'를 찾을 수 없습니다. /search 엔드포인트가 작동하지 않습니다.")
    pill_db = [] # pill_db가 None이면 오류가 나므로 빈 리스트로 초기화

# --- 비동기 처리 엔드포인트 ---

@app.route('/predict', methods=['POST'])
def predict():
    """
    (1/3) 이미지 분석을 요청하고 작업 ID를 즉시 반환 (비동기 시작점)
    이곳에서는 시간이 오래 걸리는 분석을 직접 수행하지 않음
    """
    if 'file' not in request.files:
        return jsonify({'error': '파일이 없습니다.'}), 400

    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400

    # 파일을 읽어 Base64 문자열로 인코딩하여 Celery에 전달할 준비
    filestr = file.read()
    image_string = base64.b64encode(filestr).decode('utf-8')

    # --- 여기가 핵심! ---
    # .delay()를 사용하여 Celery에게 비동기 작업을 요청
    # 작업은 Redis에 등록되고, Celery 워커가 가져가서 처리
    task = analyze_pill_image_task.delay(image_string)
    # ---------------------

    # 사용자(클라이언트)에게는 작업 ID를 즉시 반환
    # HTTP 상태 코드 202는 "요청이 성공적으로 접수되었다"는 의미
    return jsonify({'task_id': task.id}), 202

@app.route('/status/<task_id>', methods=['GET'])
def get_status(task_id):
    """
    (2/3) 작업 ID를 받아 현재 진행 상태를 확인합니다.
    클라이언트는 이 엔드포인트를 주기적으로 호출(폴링)합니다.
    """
    # Redis 백엔드에서 task_id에 해당하는 작업의 상태를 조회합니다.
    task_result = AsyncResult(id=task_id, app=analyze_pill_image_task.app)
    
    status = task_result.status
    
    # 작업 상태를 JSON 형태로 반환 (예: PENDING, SUCCESS, FAILURE)
    return jsonify({
        "task_id": task_id,
        "status": status,
    })

@app.route('/result/<task_id>', methods=['GET'])
def get_result(task_id):
    """
    (3/3) 작업이 완료되었을 때 최종 결과를 획득
    """
    task_result = AsyncResult(id=task_id, app=analyze_pill_image_task.app)

    if task_result.successful():
        # 작업이 성공적으로 완료되면, 저장된 결과(result)를 반환
        return jsonify(task_result.result)
    elif task_result.failed():
        # 작업이 실패하면, 에러 정보를 반환합니다.
        return jsonify({'error': '작업 실패', 'details': str(task_result.info)}), 500
    else:
        # 아직 작업이 진행 중인 경우, 현재 상태를 다시 공지
        return jsonify({'status': task_result.status}), 202

# --- 동기 처리 엔드포인트 ---

@app.route('/detail', methods=['GET'])
def detail():
    """
    품목기준코드를 받아 외부 API로 약물 상세 정보를 조회
    이 작업은 매우 빠르므로 동기 방식
    """
    item_code = request.args.get('code')
    if not item_code:
        return jsonify({'error': '품목기준코드가 필요합니다.'}), 400
        
    # 외부 API를 호출하여 결과를 가져옵니다.
    details = get_pill_details_from_api(item_code)
    return jsonify(details)

@app.route('/search', methods=['GET'])
def search():
    """
    (수정됨) 이름, 모양, 색, 각인을 받아 검색하고,
    API를 조회하여 이미지 URL까지 포함한 2D 리스트로 반환
    """
    # 1. 쿼리 파라미터에서 검색어 추출
    search_name = request.args.get('name', '')
    search_shape = request.args.get('shape', '')
    search_color = request.args.get('color', '')
    search_imprint = request.args.get('imprint', '')
    search_form = request.args.get('form', '')
    search_company = request.args.get('company', '')

    # 2. 검색어 유효성 검사
    if not any([search_name, search_shape, search_color, search_imprint]):
        return jsonify({'error': '하나 이상의 검색어가 필요합니다.'}), 400

    # 3. DB 로드 상태 확인
    if not pill_db:
         return jsonify({'error': '서버 DB가 로드되지 않았거나 비어있습니다.'}), 500

    # 4. database_handler로 텍스트 검색 (1차 후보군)
    initial_candidates = find_match_by_text(
        pill_db,
        name=search_name,
        shape=search_shape,
        color=search_color,
        imprint=search_imprint,
        form=search_form,         
        company=search_company   
    )

    # 5. [신규] 후보군을 순회하며 API로 이미지 URL 조회
    processed_candidates = []
    for candidate in initial_candidates:
        item_code = candidate.get('code')
        pill_name = candidate.get('pill_info')
        
        image_url = ''  # 기본값
        if item_code and item_code != 'N/A':
            try:
                # /detail에서 사용하는 API 핸들러 재사용
                api_details = get_pill_details_from_api(item_code)
                if 'error' not in api_details:
                    image_url = api_details.get('이미지', '') # '이미지' 키로 URL 가져오기
            except Exception as api_e:
                print(f"API 호출 중 에러 발생 (검색: {item_code}): {api_e}")
                image_url = 'API_ERROR'
        
        # 요청한 형식({pill_info, code, image})으로 딕셔너리 생성
        processed_candidates.append({
            'pill_info': pill_name,
            'code': item_code,
            'image': image_url
        })
    
    
    return jsonify({'pill_results': processed_candidates})

# --- 개발용 서버 실행 (Gunicorn 사용 시 이 부분은 사용되지 않음) ---
if __name__ == '__main__':
    # 이 스크립트를 직접 실행할 때 (예: python app.py) 사용하는 개발용 서버
    # 실제 운영 환경(Production)에서는 Gunicorn을 사용
    app.run(host='0.0.0.0', port=5000, debug=True)
