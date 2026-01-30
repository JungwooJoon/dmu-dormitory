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

# [1] 유틸리티 함수
def get_dynamic_score(user_score: float, criteria_str: str) -> float:
    try:
        s = float(user_score)
        mapping = {float(item.split(':')[0]): float(item.split(':')[1]) for item in criteria_str.split(',')}
        sorted_keys = sorted(mapping.keys(), reverse=True)
        for key in sorted_keys:
            if s >= key: return mapping[key]
        return 0.0
    except:
        return 0.0

def robust_to_numeric(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r'(\d+)').astype(float).fillna(0)

def parse_preference_key(pref_string: str) -> str:
    if pd.isna(pref_string): return None
    key = str(pref_string).replace('<', '').replace('>', '')
    return key.split(':', 1)[0].split('(', 1)[0].strip()

# [핵심] 개선된 중간값 경로 선택 함수 (육로 대안 우선 로직 반영)
def select_median_path(path_list: List, config: Dict, PATH_CONFIG_MAP: Dict) -> Dict:
    land_paths = []
    flight_paths = []

    # 1. 육로 대안(기차/버스) 존재 여부에 따른 그룹 분리
    for path in path_list:
        sub_paths = path.get('subPath', [])
        # 기차(4), 고속버스(5), 시외버스(6) 포함 여부 체크
        has_land = any(sub.get('trafficType') in [4, 5, 6] for sub in sub_paths)
        has_flight = any(sub.get('trafficType') == 7 for sub in sub_paths)

        if has_land:
            land_paths.append(path)
        elif has_flight:
            flight_paths.append(path)

    # 2. 분석 대상 결정 (육로가 있으면 육로만, 없으면 항공만 사용)
    target_paths = land_paths if land_paths else flight_paths
    if not target_paths:
        target_paths = path_list

    # 3. 가중 점수 계산 및 우선순위 판별
    scored_paths = []

    def get_priority_level(t_type):
        if t_type == 7: return 100  # 항공
        if t_type == 4: return 80  # 기차
        if t_type in [5, 6]: return 60  # 고속/시외버스
        return 0

    TYPE_MAP = {7: 13, 4: 11, 5: 12, 6: 12}

    for path in target_paths:
        info = path.get('info', {})
        p_type = path.get('pathType') or info.get('pathType')

        determined_type = p_type
        max_priority = -1

        # subPath 분석하여 최고 등급 수단 확정
        for sub in path.get('subPath', []):
            t_type = sub.get('trafficType')
            priority = get_priority_level(t_type)
            if priority > max_priority:
                max_priority = priority
                if t_type in TYPE_MAP:
                    determined_type = TYPE_MAP[t_type]

        config_key = PATH_CONFIG_MAP.get(determined_type)
        weight = float(config.get(config_key, 1.0)) if config_key else 1.0
        time_val = float(info.get('totalTime', 0))

        scored_paths.append({
            'path': path,
            'weighted_score': time_val * weight,
            'determined_type': determined_type,
            'applied_weight': weight
        })

    # 4. 정렬 후 중간값 반환
    scored_paths.sort(key=lambda x: x['weighted_score'])
    median_idx = len(scored_paths) // 2

    selected = scored_paths[median_idx]['path']
    selected['final_determined_type'] = scored_paths[median_idx]['determined_type']
    selected['final_applied_weight'] = scored_paths[median_idx]['applied_weight']

    return selected

async def get_kakao_coordinates(client, address, api_key):
    try:
        url = "https://dapi.kakao.com/v2/local/search/address.json"
        headers = {"Authorization": f"KakaoAK {api_key}"}
        resp = await client.get(url, headers=headers, params={"query": address}, timeout=7.0)
        data = resp.json()
        if data.get('documents'):
            return data['documents'][0]['x'], data['documents'][0]['y']
    except:
        return None, None

# [3] 메인 선발 로직
async def run_selection_process(students_df: pd.DataFrame, rooms_df: pd.DataFrame, config: Dict):
    kakao_key = config.get('카카오키', '')
    odsay_key = config.get('오디세이키', '')
    school_address = config.get('학교주소', '서울시 구로구 경인로 445')
    score_45_criteria = config.get('재학생성적', '4.5:30,4.0:25,3.5:20,3.0:15,2.5:10')
    score_1000_criteria = config.get('신입생성적', '950:30,900:25,850:20,800:15,750:10,700:5')

    PATH_NAME_MAP = {1: "지하철", 2: "버스", 3: "버스+지하철", 11: "열차(KTX/SRT)", 12: "고속/시외버스", 13: "항공", 20: "시외교통 복합"}
    PATH_CONFIG_MAP = {13: "항공페널티", 11: "고속철도페널티", 20: "고속철도페널티", 12: "고속버스페널티", 2: "광역버스페널티"}

    # 방 정보 세팅
    rooms_df['capacity'] = robust_to_numeric(rooms_df['인실'])
    rooms_df['Type_Key'] = rooms_df['유형'].apply(parse_preference_key)
    room_price_map = rooms_df.drop_duplicates(subset=['Type_Key']).set_index('Type_Key')['가격'].to_dict()

    cap_grouped = rooms_df.groupby(['성별', 'Type_Key'])['capacity'].sum()
    female_cap = cap_grouped.loc['여자'].to_dict() if '여자' in cap_grouped.index else {}
    male_cap = cap_grouped.loc['남자'].to_dict() if '남자' in cap_grouped.index else {}

    # 거리 계산 루프
    async with httpx.AsyncClient() as client:
        school_coords = await get_kakao_coordinates(client, school_address, kakao_key)
        raw_commute_scores, commute_times, transport_modes, applied_weights = [], [], [], []

        for idx, row in students_df.iterrows():
            # 학생 양식 '집주소' 컬럼 사용
            addr = row.get('집주소', '')
            x, y = await get_kakao_coordinates(client, addr, kakao_key)
            score, time_val, mode_val, weight_val = 0.0, 0, "-", 1.0

            if x and y and school_coords[0]:
                try:
                    url = "https://api.odsay.com/v1/api/searchPubTransPathT"
                    params = {"apiKey": odsay_key, "SX": x, "SY": y, "EX": school_coords[0], "EY": school_coords[1]}
                    resp = await client.get(url, params=params, timeout=7.0)
                    data = resp.json()

                    if data and "result" in data and data['result'].get('path'):
                        rdata = select_median_path(data['result']['path'], config, PATH_CONFIG_MAP)
                        info = rdata.get('info', {})

                        final_type = rdata.get('final_determined_type')
                        weight_val = rdata.get('final_applied_weight', 1.0)
                        mode_val = PATH_NAME_MAP.get(final_type, "대중교통")

                        try:
                            time_val = int(info.get('totalTime', 0))
                        except:
                            time_val = 0

                        score = float(time_val) * weight_val
                except Exception as e:
                    logger.error(f"분석 오류 ({addr}): {e}")

            # 리스트 개수 정렬을 위해 루프 끝에서 한 번만 append
            raw_commute_scores.append(score)
            commute_times.append(time_val)
            transport_modes.append(mode_val)
            applied_weights.append(weight_val)
            await asyncio.sleep(0.1)

    # 점수 저장 및 환산
    max_raw = max(raw_commute_scores) if raw_commute_scores else 1.0
    students_df['통학 점수(70점)'] = [(s / max_raw) * 70 for s in raw_commute_scores]
    students_df['성적 점수(30점)'] = students_df['성적'].apply(
        lambda x: get_dynamic_score(float(x or 0), score_1000_criteria if float(x or 0) > 5 else score_45_criteria))
    students_df['최종 점수'] = students_df['통학 점수(70점)'] + students_df['성적 점수(30점)']
    students_df['소요시간(분)'] = commute_times
    students_df['교통수단'] = transport_modes
    students_df['적용 가중치'] = applied_weights

    # 5. 배정 데이터 초기화 및 키 생성
    for col in ['배정결과', '배정방식', '배정된 방']: students_df[col] = '-'
    for i in range(1, 4):
        col_name = f'{i}지망'
        if col_name in students_df.columns:
            students_df[f'{i}지망_Key'] = students_df[col_name].apply(parse_preference_key)

    # 6. 배정 프로세스
    pri_mask = students_df['우선선발'].isin(['O', True, 'true', '1'])
    # 우선선발 대상자 배정
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

    # 일반 선발 (지망 순위별 루프)
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

    # 지망에서 탈락한 인원 임의 배정
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