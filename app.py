import base64
from flask import Flask, request, jsonify, session
from flask_session import Session
import os
from dotenv import load_dotenv
from celery.result import AsyncResult
from datetime import timedelta
import redis

from flask import send_file
import io
import requests

from tasks import analyze_pill_image_task
from api_handler import get_pill_details_from_api, get_pill_image_from_api
from database_handler import load_database, find_match_by_text

load_dotenv()

app = Flask(__name__)

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")
app.config["SESSION_TYPE"] = "redis"
app.config["SESSION_PERMANENT"] = True # 세션 유지
app.config["SESSION_USE_SIGNER"] = True # 세션 쿠키를 암호화
app.config["SESSION_REDIS"] = redis.Redis(host='localhost', port=6379, db=1) # db=1 사용
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=1)
app.config["SESSION_REFRESH_EACH_REQUEST"] = True

Session(app)

try:
    redis_client = redis.Redis(host='localhost', port=6379, db=2, decode_responses=True)
    redis_client.ping()
    print("Redis (db=2)에 성공적으로 연결되었습니다.")
except Exception as e:
    print(f"경고: Redis (db=2) 연결에 실패했습니다. {e}") # 에러
    redis_client = None

RECENT_PILLS_KEY_PREFIX = "recent_pills:session:" # 키 접두사
RECENT_PILLS_MAX_COUNT = 5

DB_PATH = 'database/pill.csv'
pill_db = load_database(DB_PATH)
if pill_db is None:
    print(f"경고: '{DB_PATH}'를 찾을 수 없습니다. /search 엔드포인트가 작동하지 않습니다.")
    pill_db = [] # pill_db가 None이면 오류가 나므로 빈 리스트로 초기화

@app.route('/image-proxy')
def image_proxy():
# 이미지 다운로드 URL을 대신 받아서, Content-Disposition 헤더를 제거하고 이미지로 스트리밍.
    
    image_url = request.args.get('url')
    if not image_url:
        return "URL이 필요합니다.", 400
    
    try:
        r = requests.get(image_url)
        r.raise_for_status() # HTTP 에러 체크

        # 원본의 콘텐츠 타입을 그대로 사용
        content_type = r.headers.get('Content-Type', 'image/jpeg')

        # 다운로드한 이미지의 바이트 데이터를 파일처럼 전송
        return send_file(
            io.BytesIO(r.content),
            mimetype=content_type,
            as_attachment=False
        )
    except Exception as e:
        print(f"이미지 프록시 실패: {e}")
        return "이미지를 찾을 수 없습니다.", 404
    
    
# --- 비동기 처리 엔드포인트 ---
@app.route('/predict', methods=['POST'])
def predict():
#이미지 분석을 요청하고 작업 ID를 즉시 반환

    if 'file' not in request.files:
        return jsonify({'error': '파일이 없습니다.'}), 400

    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400

    # 파일을 읽어 Base64 문자열로 인코딩하여 Celery에 전달할 준비
    filestr = file.read()
    image_string = base64.b64encode(filestr).decode('utf-8')

    # .delay()를 사용하여 Celery에게 비동기 작업을 요청
    # 작업은 Redis에 등록되고, Celery 워커가 가져가서 처리
    task = analyze_pill_image_task.delay(image_string)

    # 사용자(클라이언트)에게는 작업 ID를 즉시 반환
    return jsonify({'task_id': task.id}), 202

@app.route('/status/<task_id>', methods=['GET'])
def get_status(task_id):
# 작업 ID를 받아 현재 진행 상태를 확인

    # Redis 백엔드에서 task_id에 해당하는 작업의 상태를 조회
    task_result = AsyncResult(id=task_id, app=analyze_pill_image_task.app)
    
    status = task_result.status
    
    # 작업 상태를 JSON 형태로 반환 (예: PENDING, SUCCESS, FAILURE)
    return jsonify({
        "task_id": task_id,
        "status": status,
    })

@app.route('/result/<task_id>', methods=['GET'])
def get_result(task_id):
# 작업이 완료되었을 때 최종 결과를 획득

    task_result = AsyncResult(id=task_id, app=analyze_pill_image_task.app)

    if task_result.successful():
        # 작업이 성공적으로 완료되면, 저장된 결과를 반환
        return jsonify(task_result.result)
    elif task_result.failed():
        # 작업이 실패하면, 에러 정보를 반환
        return jsonify({'error': '작업 실패', 'details': str(task_result.info)}), 500
    else:
        # 아직 작업이 진행 중인 경우, 현재 상태를 다시 공지
        return jsonify({'status': task_result.status}), 202

# --- 동기 처리 엔드포인트 ---
@app.route('/detail', methods=['GET'])
def detail():
# 품목기준코드를 받아 외부 API로 약물 상세 정보를 조회

    item_code = request.args.get('code')
    if not item_code:
        return jsonify({'error': '품목기준코드가 필요합니다.'}), 400
        
    # 외부 API를 호출하여 결과 추출
    details = get_pill_details_from_api(item_code)
    
    if 'error' not in details and redis_client and session.sid:
       try:
           session['last_viewed_code'] = item_code
            
           # 세션 ID를 사용하여 사용자별 고유 키 생성
           user_key = f"{RECENT_PILLS_KEY_PREFIX}{session.sid}:list"

           redis_client.lrem(user_key, 0, item_code)
           redis_client.lpush(user_key, item_code)
           redis_client.ltrim(user_key, 0, RECENT_PILLS_MAX_COUNT - 1)
           
       except Exception as e:
           print(f"Redis 작업 실패 (/detail, session: {session.sid}): {e}")
           
    return jsonify(details)

@app.route('/search', methods=['GET'])
def search():
# 검색어로 DB를 검색하고, 페이지네이션을 적용하여 반환

    # 쿼리 파라미터에서 검색어 추출
    search_name = request.args.get('name', '')
    search_shape = request.args.get('shape', '')
    search_color = request.args.get('color', '')
    search_imprint = request.args.get('imprint', '')
    search_form = request.args.get('form', '')
    search_company = request.args.get('company', '')

    # 페이지네이션 파라미터 추출
    try:
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 10)) # 한 페이지에 20개씩
    except ValueError:
        page = 1
        limit = 10
    
    if page < 1: page = 1
    if limit < 1: limit = 10

    # DB 로드 상태 확인
    if not pill_db:
         return jsonify({'error': '서버 DB가 로드되지 않았거나 비어있습니다.'}), 500

    # database_handler로 텍스트 검색 (전체 후보군)
    initial_candidates = find_match_by_text(
        pill_db,
        name=search_name,
        shape=search_shape,
        color=search_color,
        imprint=search_imprint,
        form=search_form,
        company=search_company
    )

    # 페이지네이션 적용
    total_items = len(initial_candidates)
    start_index = (page - 1) * limit
    end_index = page * limit
    
    # 전체 후보군에서 현재 페이지에 해당하는 20개만 잘라냄
    paginated_candidates = initial_candidates[start_index:end_index]

    # 잘라낸 20개에 대해서만 정보 조회
    processed_candidates = []
    for candidate in paginated_candidates:
        item_code = candidate.get('code')
        pill_name = candidate.get('pill_info')
        image_url = candidate.get('image_url', '')
        
        processed_candidates.append({
            'pill_info': pill_name,
            'code': item_code,
            'image': image_url
        })
    
    # 페이지네이션 정보와 함께 결과 반환
    return jsonify({
        'total_items': total_items,
        'page': page,
        'limit': limit,
        'total_pages': (total_items + limit - 1) // limit,
        'pill_results': processed_candidates
    })

@app.route('/recent', methods=['GET'])
def get_recent():
# 해당 세션의 최근 검색 목록을 반환

    if not redis_client:
        return jsonify({'error': 'Redis에 연결되지 않았습니다.'}), 500

    # 세션 ID가 없으면 빈 리스트 반환
    if not session.sid:
        return jsonify({'pill_results': []}) # ID가 없으면 그냥 빈 내역

    pill_results = []
    
    try:
        # 세션 ID를 사용하여 사용자별 고유 키로 조회
        user_key = f"{RECENT_PILLS_KEY_PREFIX}{session.sid}:list"
        item_codes = redis_client.lrange(user_key, 0, -1)

        # 정보 조회
        for code in item_codes:
            details = get_pill_image_from_api(code)

            if 'error' not in details:
                pill_results.append({
                    'pill_info': details.get('제품명', '이름 없음'),
                    'code': code,
                    'image': details.get('이미지', '')
                })

    except Exception as e:
        print(f"Redis 작업 실패 (/recent, session: {session.sid}): {e}")
        return jsonify({'error': '최근 검색 목록을 불러오는 중 오류 발생'}), 500

    return jsonify({'pill_results': pill_results})

@app.after_request
def refresh_recent_list_expiration(response):
# 세션이 있고, 최근 내역 데이터가 존재하면 만료 시간을 갱신

    try:
        # SESSION_REFRESH_EACH_REQUEST가 True이고, Redis(db=2)가 연결되어 있고, 세션 ID가 있을 때만 실행
        if app.config.get("SESSION_REFRESH_EACH_REQUEST") and redis_client and session.sid:
            
            # 사용자별 최근 내역 키 생성
            user_key = f"{RECENT_PILLS_KEY_PREFIX}{session.sid}:list"
            
            # 해당 키가 실제로 Redis(db=2)에 존재하는지 확인
            if redis_client.exists(user_key):
                
                # 설정된 세션 만료 시간 가져오기
                session_lifetime_seconds = app.config.get('PERMANENT_SESSION_LIFETIME', timedelta(days=1)).total_seconds()
                
                # 키 만료 시간 갱신
                redis_client.expire(user_key, int(session_lifetime_seconds))
                
    except Exception as e:
        print(f"Error refreshing recent list expiration: {e}")
        
    return response


# --- 개발용 서버 실행 (Gunicorn 사용 시 이 부분은 사용되지 않음) ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

