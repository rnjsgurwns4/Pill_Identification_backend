
import requests
import xml.etree.ElementTree as ET
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# 공공데이터포털에서 발급받은 일반 인증키를 입력
API_KEY=os.getenv("GO_DATA_API_KEY").strip()
# --------------------------------------------------------------------

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
        return details

    except requests.exceptions.RequestException as e:
        return {'error': f'API 요청 실패: {e}'}
    except ET.ParseError as e:
        return {'error': f'XML 파싱 실패: {e}'}

