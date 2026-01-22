import pandas as pd
import httpx
import asyncio
import logging
import numpy as np
import io
from typing import Dict, List, Tuple, Optional

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SelectionLogic")


# [1] 유틸리티 함수 (기존 유지)
def get_dynamic_score(user_score: float, criteria_str: str) -> float:
    try:
        s = float(user_score)
        mapping = {float(item.split(':')[0]): float(item.split(':')[1]) for item in criteria_str.split(',')}
        sorted_keys = sorted(mapping.keys(), reverse=True)
        for key in sorted_keys:
            if s >= key: return mapping[key]
        return 0.0
    except Exception as e:
        logger.error(f"성적 환산 오류: {e}")
        return 0.0


def robust_to_numeric(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r'(\d+)').astype(float).fillna(0)


def parse_preference_key(pref_string: str) -> str:
    if pd.isna(pref_string): return None
    key = str(pref_string).replace('<', '').replace('>', '')
    return key.split(':', 1)[0].split('(', 1)[0].strip()


# [개선] 경로 선택 로직 - 데이터 누락 방지 강화
def select_best_path(path_list: List, config: Dict, PATH_CONFIG_MAP: Dict) -> Dict:
    # 1단계: 기본값으로 첫 번째 경로 설정
    best_selected = path_list[0]
    max_path_score = -1.0

    for path in path_list:
        # rdata 직속 또는 info 내부에서 pathType 추출
        info = path.get('info', {})
        p_type = path.get('pathType') or info.get('pathType')

        # 소요 시간 추출 (안전하게 int 변환)
        try:
            time_val = float(info.get('totalTime', 0))
        except:
            time_val = 0.0

        # 가중치 확인 (설정 파일에 없으면 기본 1.0)
        config_key = PATH_CONFIG_MAP.get(p_type)
        weight = float(config.get(config_key, 1.0)) if config_key else 1.0

        current_path_score = time_val * weight

        # 가장 높은 점수를 주는 경로를 선택
        if current_path_score > max_path_score:
            max_path_score = current_path_score
            best_selected = path

    return best_selected


# [2] 비동기 API 호출
async def get_kakao_coordinates(client: httpx.AsyncClient, address: str, api_key: str):
    try:
        url = "https://dapi.kakao.com/v2/local/search/address.json"
        headers = {"Authorization": f"KakaoAK {api_key}"}
        resp = await client.get(url, headers=headers, params={"query": address}, timeout=7.0)
        data = resp.json()
        if data.get('documents'):
            return data['documents'][0]['x'], data['documents'][0]['y']
    except Exception as e:
        logger.error(f"카카오 API 오류: {address} - {str(e)}")
    return None, None


# [3] 메인 선발 로직
async def run_selection_process(students_df: pd.DataFrame, rooms_df: pd.DataFrame, config: Dict):
    logger.info(f"입사생 선발 알고리즘 가동: 학생 {len(students_df)}명 분석")

    # 설정 로드
    kakao_key = config.get('카카오키', '')
    odsay_key = config.get('오디세이키', '')
    school_address = config.get('학교주소', '서울시 구로구 경인로 445')
    score_45_criteria = config.get('재학생성적', '4.5:30,4.0:25,3.5:20,3.0:15,2.5:10')
    score_1000_criteria = config.get('신입생성적', '950:30,900:25,850:20,800:15,750:10,700:5')

    PATH_NAME_MAP = {
        1: "지하철", 2: "버스", 3: "버스+지하철",
        11: "열차(KTX/SRT)", 12: "고속/시외버스", 13: "항공",
        20: "시외교통 복합"
    }
    PATH_CONFIG_MAP = {
        13: "항공페널티", 11: "고속철도페널티", 20: "고속철도페널티",
        12: "고속버스페널티", 2: "광역버스페널티"
    }

    # 방 정보 세팅
    rooms_df['capacity'] = robust_to_numeric(rooms_df['인실'])
    rooms_df['Type_Key'] = rooms_df['유형'].apply(parse_preference_key)
    room_price_map = rooms_df.drop_duplicates(subset=['Type_Key']).set_index('Type_Key')['가격'].to_dict()

    cap_grouped = rooms_df.groupby(['성별', 'Type_Key'])['capacity'].sum()
    female_cap = cap_grouped.loc['여자'].to_dict() if '여자' in cap_grouped.index else {}
    male_cap = cap_grouped.loc['남자'].to_dict() if '남자' in cap_grouped.index else {}

    # 거리 계산
    async with httpx.AsyncClient() as client:
        school_coords = await get_kakao_coordinates(client, school_address, kakao_key)
        raw_commute_scores, commute_times, transport_modes = [], [], []

        for idx, row in students_df.iterrows():
            addr = row['주소']
            x, y = await get_kakao_coordinates(client, addr, kakao_key)
            score, time_val, mode_val = 0.0, 0, "-"

            if x and y and school_coords[0]:
                try:
                    url = "https://api.odsay.com/v1/api/searchPubTransPathT"
                    params = {"apiKey": odsay_key, "SX": x, "SY": y, "EX": school_coords[0], "EY": school_coords[1]}
                    resp = await client.get(url, params=params, timeout=7.0)
                    data = resp.json()

                    if "result" in data and data['result'].get('path'):
                        # 최적 경로 선택 함수 호출
                        rdata = select_best_path(data['result']['path'], config, PATH_CONFIG_MAP)

                        info = rdata.get('info', {})
                        # 소요 시간 추출 시도
                        try:
                            time_val = int(info.get('totalTime', 0))
                        except:
                            time_val = 0

                        # pathType 추출 보강 (rdata 직속 -> info 내부)
                        p_type = rdata.get('pathType') or info.get('pathType')

                        if p_type in PATH_NAME_MAP:
                            mode_val = PATH_NAME_MAP[p_type]
                            config_key = PATH_CONFIG_MAP.get(p_type)
                            final_weight = float(config.get(config_key, 1.0)) if config_key else 1.0
                        else:
                            mode_val = f"기타({p_type})" if p_type else "대중교통"
                            final_weight = 1.0

                        score = float(time_val) * final_weight
                except Exception as e:
                    logger.error(f"경로 분석 누락 ({addr}): {e}")

            raw_commute_scores.append(score)
            commute_times.append(time_val)
            transport_modes.append(mode_val)
            # API 초과 방지를 위해 대기 시간 약간 상향
            await asyncio.sleep(0.1)

    # 점수 환산 및 배정 로직 (기존과 동일)
    max_raw = max(raw_commute_scores) if raw_commute_scores else 1.0
    students_df['통학 점수(70점)'] = [(s / max_raw) * 70 for s in raw_commute_scores]
    students_df['성적 점수(30점)'] = students_df['성적'].apply(
        lambda x: get_dynamic_score(float(x), score_1000_criteria if float(x) > 5 else score_45_criteria))
    students_df['최종 점수'] = students_df['통학 점수(70점)'] + students_df['성적 점수(30점)']
    students_df['소요시간(분)'] = commute_times
    students_df['교통수단'] = transport_modes

    # 5. 배정 데이터 초기화 및 키 생성
    for col in ['배정결과', '배정방식', '배정된 방']: students_df[col] = '-'
    for i in range(1, 4):
        col_name = f'{i}지망'
        if col_name in students_df.columns:
            students_df[f'{i}지망_Key'] = students_df[col_name].apply(parse_preference_key)

    # 6. 배정 프로세스 (기존 로직 유지)
    pri_mask = students_df['우선선발'].isin(['O', True, 'true', '1'])
    for idx in students_df[pri_mask].sort_values(by='최종 점수', ascending=False).index:
        std = students_df.loc[idx]
        cmap = female_cap if std['성별'] == '여자' else male_cap
        assigned = False
        for i in range(1, 4):
            ck = std.get(f'{i}지망_Key')
            if ck in cmap and cmap[ck] > 0:
                students_df.loc[idx, ['배정결과', '배정된 방', '배정방식']] = ['합격 (우선)', ck, f'{i}지망 (우선)']
                cmap[ck] -= 1
                assigned = True
                break
        if not assigned:
            for r, s in cmap.items():
                if s > 0:
                    students_df.loc[idx, ['배정결과', '배정된 방', '배정방식']] = ['합격 (우선)', r, '임의 (우선)']
                    cmap[r] -= 1
                    break

    for i in range(1, 4):
        target_key = f'{i}지망_Key'
        unassigned_mask = ~pri_mask & (students_df['배정결과'] == '-')
        sorted_indices = students_df[unassigned_mask].sort_values(by='최종 점수', ascending=False).index
        for idx in sorted_indices:
            ck = students_df.loc[idx, target_key]
            gen = students_df.loc[idx, '성별']
            cmap = female_cap if gen == '여자' else male_cap
            if ck in cmap and cmap[ck] > 0:
                students_df.loc[idx, ['배정결과', '배정된 방', '배정방식']] = ['합격 (일반)', ck, f'{i}지망']
                cmap[ck] -= 1

    final_mask = (students_df['배정결과'] == '-')
    for idx in students_df[final_mask].index:
        gen = students_df.loc[idx, '성별']
        cmap = female_cap if gen == '여자' else male_cap
        for r, s in cmap.items():
            if s > 0:
                students_df.loc[idx, ['배정결과', '배정된 방', '배정방식']] = ['합격 (일반)', r, '임의 배정']
                cmap[r] -= 1
                break
        else:
            students_df.loc[idx, '배정결과'] = '불합격(T.O부족)'
            students_df.loc[idx, '배정방식'] = '예비 순번'

    return students_df, room_price_map
