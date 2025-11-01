
import requests
import xml.etree.ElementTree as ET
import os
from dotenv import load_dotenv
import base64
# .env 파일에서 환경 변수 로드
load_dotenv()

# 공공데이터포털에서 발급받은 일반 인증키를 입력
API_KEY=os.getenv("GO_DATA_API_KEY").strip()
# --------------------------------------------------------------------

def _get_pill_identification_info(item_code):
    """
    (헬퍼) 낱알식별정보 API를 호출하여 장축, 단축, 두께, 성상 정보를 가져옵니다.
    """
    if not API_KEY:
        return {'error': '.env 파일에 GO_DATA_API_KEY를 설정해주세요.'}

    # 낱알식별정보 API 서비스 URL
    url = 'http://apis.data.go.kr/1471000/MdcinGrnIdntfcInfoService02/getMdcinGrnIdntfcInfoList02'
    params = {
        'ServiceKey': API_KEY,
        'itemSeq': item_code,
        'type': 'xml'
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        
        result_code_element = root.find('header/resultCode')
        if result_code_element is None or result_code_element.text != '00':
            # 낱알식별정보가 없는 경우, 에러가 아닌 빈 딕셔너리 반환
            return {} 

        item = root.find('body/items/item')
        if item is None:
            return {} # 정보가 없으면 빈 딕셔너리 반환

        # 필요한 정보(장축, 단축, 두께, 성상) 추출
        id_info = {
            '장축': item.find('LENG_LONG').text if item.find('LENG_LONG') is not None else '',
            '단축': item.find('LENG_SHORT').text if item.find('LENG_SHORT') is not None else '',
            '두께': item.find('THICK').text if item.find('THICK') is not None else '',
            '각인_1': item.find('PRINT_FRONT').text if item.find('PRINT_FRONT') is not None else '',
            '각인_2': item.find('PRINT_BACK').text if item.find('PRINT_BACK') is not None else '',
            '색_1': item.find('COLOR_CLASS1').text if item.find('COLOR_CLASS2') is not None else '',
            '색_2': item.find('COLOR_CLASS1').text if item.find('COLOR_CLASS2') is not None else '',
            '모양': item.find('DRUG_SHAPE').text if item.find('DRUG_SHAPE') is not None else '',
            '형태': item.find('FORM_CODE_NAME').text if item.find('FORM_CODE_NAME') is not None else '',
        }
        return id_info

    except Exception as e:
        # 이 API 호출이 실패해도 메인 기능은 동작해야 하므로, 에러 대신 빈 딕셔너리 반환
        print(f"낱알식별정보 API 호출 실패 (item_code: {item_code}): {e}")
        return {}
    

def encode_image_from_url_to_base64(image_url):
    """
    이미지 다운로드 URL을 받아, 이미지를 다운로드하고
    Base64 데이터 URL 문자열로 반환합니다.
    <img src="...">에 바로 사용할 수 있습니다.
    """
    if not image_url:
        return ""
    
    try:
        response = requests.get(image_url)
        response.raise_for_status() # 에러가 있으면 예외 발생

        # MIME 타입 확인 (예: 'image/jpeg')
        mime_type = response.headers.get('Content-Type', 'image/jpeg')
        
        # Base64로 인코딩
        base64_data = base64.b64encode(response.content).decode('utf-8')
        
        # 데이터 URL 형식으로 반환
        return f"data:{mime_type};base64,{base64_data}"
        
    except requests.exceptions.RequestException as e:
        print(f"이미지 다운로드 실패 (URL: {image_url}): {e}")
        return "" # 실패 시 빈 문자열 반환

def get_pill_details_from_api(item_code):
    """
    공공데이터포털 API를 호출하여 품목기준코드로 알약 상세 정보를 조회
    """
    if not API_KEY:
        return {'error': '.env 파일에 GO_DATA_API_KEY를 설정해주세요.'}

    url = 'http://apis.data.go.kr/1471000/DrbEasyDrugInfoService/getDrbEasyDrugList'
    params = {
        'ServiceKey': API_KEY,
        'itemSeq': item_code,
        'type': 'xml' 
    }

    try:
        
        # 요청 빋기
        response = requests.get(url, params=params)
        response.raise_for_status() # HTTP 오류 발생 시 예외 처리

        # XML 응답 파싱
        root = ET.fromstring(response.content)
        xml_content = ET.tostring(root, encoding='utf-8').decode('utf-8')
        print(xml_content)
        
        # header에서 결과 코드 확인
        result_code_element = root.find('header/resultCode')
        if result_code_element is not None:
            result_code = result_code_element.text
        else:
            # resultCode 태그를 찾지 못했을 때의 처리
            return {'error': 'API 응답 형식이 올바르지 않습니다.'}
        if result_code != '00':
            return {'error': f"API 오류: {root.find('header/resultMsg').text}"}
            
        item = root.find('body/items/item')
        if item is None:
            return {'error': '해당 품목기준코드에 대한 정보가 없습니다.'}

        # 필요한 정보만 추출하여 딕셔너리로 반환 (None일 경우 빈 문자열 처리)
        details = {
            '제품명': item.find('itemName').text if item.find('itemName') is not None else '',
            '업체명': item.find('entpName').text if item.find('entpName') is not None else '',
            '효능': item.find('efcyQesitm').text if item.find('efcyQesitm') is not None else '',
            '사용법': item.find('useMethodQesitm').text if item.find('useMethodQesitm') is not None else '',
            '주의사항경고': item.find('atpnWarnQesitm').text if item.find('atpnWarnQesitm') is not None else '',
            '주의사항': item.find('atpnQesitm').text if item.find('atpnQesitm') is not None else '',
            '부작용': item.find('seQesitm').text if item.find('seQesitm') is not None else '',
            '보관법': item.find('depositMethodQesitm').text if item.find('depositMethodQesitm') is not None else '',
            '이미지': item.find('itemImage').text if item.find('itemImage') is not None else ''
        }
    

    except requests.exceptions.RequestException as e:
        return {'error': f'API 요청 실패: {e}'}
    except ET.ParseError as e:
        return {'error': f'XML 파싱 실패: {e}'}
    
    id_info = _get_pill_identification_info(item_code)
    
    # 3. (신규) 두 딕셔너리 병합
    # id_info의 내용을 details 딕셔너리에 추가합니다.
    details.update(id_info)
    
    # 4. (수정) 병합된 딕셔너리 반환
    return details
    
def get_pill_image_from_api(item_code):
    """
    공공데이터포털 API를 호출하여 품목기준코드로 알약 상세 정보를 조회
    """
    if not API_KEY:
        return {'error': '.env 파일에 GO_DATA_API_KEY를 설정해주세요.'}

    url = 'http://apis.data.go.kr/1471000/DrbEasyDrugInfoService/getDrbEasyDrugList'
    params = {
        'ServiceKey': API_KEY,
        'itemSeq': item_code,
        'type': 'xml' 
    }

    try:
        
        # 요청 빋기
        response = requests.get(url, params=params)
        response.raise_for_status() # HTTP 오류 발생 시 예외 처리

        # XML 응답 파싱
        root = ET.fromstring(response.content)
        xml_content = ET.tostring(root, encoding='utf-8').decode('utf-8')
        print(xml_content)
        
        # header에서 결과 코드 확인
        result_code_element = root.find('header/resultCode')
        if result_code_element is not None:
            result_code = result_code_element.text
        else:
            # resultCode 태그를 찾지 못했을 때의 처리
            return {'error': 'API 응답 형식이 올바르지 않습니다.'}
        if result_code != '00':
            return {'error': f"API 오류: {root.find('header/resultMsg').text}"}
            
        item = root.find('body/items/item')
        if item is None:
            return {'error': '해당 품목기준코드에 대한 정보가 없습니다.'}

        # 필요한 정보만 추출하여 딕셔너리로 반환 (None일 경우 빈 문자열 처리)
        details = {
            '제품명': item.find('itemName').text if item.find('itemName') is not None else '',
            '이미지': item.find('itemImage').text if item.find('itemImage') is not None else ''
        }
        return details

    except requests.exceptions.RequestException as e:
        return {'error': f'API 요청 실패: {e}'}
    except ET.ParseError as e:
        return {'error': f'XML 파싱 실패: {e}'}

