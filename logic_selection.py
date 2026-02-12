import pandas as pd
import httpx
import asyncio
import logging
import numpy as np
import io
import math
from urllib.parse import quote
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SelectionLogic")


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


async def get_city_travel_time(client: httpx.AsyncClient, odsay_key: str, sx, sy, ex, ey) -> int:
    try:
        url = "https://api.odsay.com/v1/api/searchPubTransPathT"
        params = {"apiKey": odsay_key, "SX": sx, "SY": sy, "EX": ex, "EY": ey, "searchType": 0}
        resp = await client.get(url, params=params, timeout=5.0)
        data = resp.json()
        if data and "result" in data and data['result'].get('path'):
            return int(data['result']['path'][0]['info'].get('totalTime', 0))
    except:
        pass
    return 15


async def select_refined_path(client, odsay_key, path_list, config, PATH_CONFIG_MAP, student_coords):
    scored_paths = []
    TYPE_MAP = {7: 13, 4: 11, 5: 12, 6: 12}

    first_path = path_list[0]
    info_ref = first_path.get('info', {})
    p_type_ref = first_path.get('pathType')
    is_intercity = (p_type_ref in [11, 12, 13, 20]) or (info_ref.get('trafficDistance', 0) > 50000)

    t1_val = 0
    start_station_name = ""
    if is_intercity:
        sub_paths_ref = first_path.get('subPath', [])
        first_transit = next((s for s in sub_paths_ref if s.get('trafficType') in [1, 2, 4, 5, 6, 7]), None)
        if first_transit:
            start_station_name = first_transit.get('startName', '')
            t1_val = await get_city_travel_time(client, odsay_key, student_coords[0], student_coords[1],
                                                first_transit['startX'], first_transit['startY'])

    for path in path_list:
        info = path.get('info', {})
        pure_time = int(info.get('totalTime', 0))
        sub_paths = path.get('subPath', [])

        full_time = pure_time + t1_val
        route_parts = []

        if t1_val > 0:
            route_parts.append(f"집({t1_val}분) → {start_station_name}")

        for s in sub_paths:
            t_type = s.get('trafficType')
            if t_type in [1, 2, 4, 5, 6, 7]:
                lane = s.get('lane', [{}])[0]
                line = lane.get('busNo') or lane.get('name') or "이동"
                s_name = s.get('startName', '')
                e_name = s.get('endName', '')
                route_parts.append(f"{s_name}[{line}] → {e_name}")

        final_route = " → ".join(route_parts)

        traffic_types = [s.get('trafficType') for s in sub_paths]
        det_type = path.get('pathType')
        if 7 in traffic_types:
            det_type = 13
        elif 4 in traffic_types:
            det_type = 11
        elif any(t in traffic_types for t in [5, 6]):
            det_type = 12

        weight = float(config.get(PATH_CONFIG_MAP.get(det_type), 1.0))
        scored_paths.append({
            'weighted_score': full_time * weight,
            'final_time': full_time,
            'determined_type': det_type,
            'applied_weight': weight,
            'route_summary': final_route
        })

    if not scored_paths: return None
    scored_paths.sort(key=lambda x: x['weighted_score'])
    return scored_paths[0]


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


async def run_selection_process(students_df: pd.DataFrame, rooms_df: pd.DataFrame, config: Dict):
    kakao_key = config.get('카카오키', '')
    odsay_key = config.get('오디세이키', '')
    school_address = config.get('학교주소', '서울시 구로구 경인로 445')
    score_45_criteria = config.get('재학생성적', '4.5:30,4.0:25,3.5:20,3.0:15,2.5:10')
    score_1000_criteria = config.get('신입생성적', '950:30,900:25,850:20,800:15,750:10,700:5')

    PATH_NAME_MAP = {1: "지하철", 2: "버스", 3: "버스+지하철", 11: "열차(KTX/SRT)", 12: "고속/시외버스", 13: "항공", 20: "시외교통 복합"}
    PATH_CONFIG_MAP = {13: "항공페널티", 11: "고속철도페널티", 20: "고속철도페널티", 12: "고속버스페널티", 2: "광역버스페널티"}

    rooms_df['capacity'] = robust_to_numeric(rooms_df['인실'])
    rooms_df['Type_Key'] = rooms_df['유형'].apply(parse_preference_key)
    room_price_map = rooms_df.drop_duplicates(subset=['Type_Key']).set_index('Type_Key')['가격'].to_dict()

    cap_grouped = rooms_df.groupby(['성별', 'Type_Key'])['capacity'].sum()
    female_cap = cap_grouped.loc['여자'].to_dict() if '여자' in cap_grouped.index else {}
    male_cap = cap_grouped.loc['남자'].to_dict() if '남자' in cap_grouped.index else {}

    async with httpx.AsyncClient() as client:
        sx, sy = await get_kakao_coordinates(client, school_address, kakao_key)
        results = []

        for idx, row in students_df.iterrows():
            addr = row.get('집주소', '')
            x, y = await get_kakao_coordinates(client, addr, kakao_key)
            s_val, t_val, m_val, w_val, r_detail = 0.0, 0, "-", 1.0, "경로없음"

            if x and y and sx:
                try:
                    url = "https://api.odsay.com/v1/api/searchPubTransPathT"
                    params = {"apiKey": odsay_key, "SX": x, "SY": y, "EX": sx, "EY": sy}
                    resp = await client.get(url, params=params, timeout=7.0)
                    data = resp.json()

                    if data and "result" in data and data['result'].get('path'):
                        rdata = await select_refined_path(client, odsay_key, data['result']['path'], config,
                                                          PATH_CONFIG_MAP, (x, y))
                        if rdata:
                            t_val = rdata['final_time']
                            w_val = rdata['applied_weight']
                            m_val = PATH_NAME_MAP.get(rdata['determined_type'], "대중교통")
                            s_val = float(t_val) * w_val
                            r_detail = rdata['route_summary']
                except Exception as e:
                    logger.error(f"Error for {addr}: {e}")

            results.append({'s': s_val, 't': t_val, 'm': m_val, 'w': w_val, 'r': r_detail})
            await asyncio.sleep(0.5)

    students_df['소요시간(분)'] = [r['t'] for r in results]
    students_df['교통수단'] = [r['m'] for r in results]
    students_df['상세경로'] = [r['r'] for r in results]
    students_df['적용 가중치'] = [r['w'] for r in results]

    max_raw = max([r['s'] for r in results]) if any([r['s'] for r in results]) else 1.0
    students_df['통학 점수(70점)'] = [(r['s'] / max_raw) * 70 for r in results]
    students_df['성적 점수(30점)'] = students_df['성적'].apply(
        lambda x: get_dynamic_score(float(x or 0), score_1000_criteria if float(x or 0) > 5 else score_45_criteria))
    students_df['최종 점수'] = students_df['통학 점수(70점)'] + students_df['성적 점수(30점)']

    for col in ['배정결과', '배정방식', '배정된 방']: students_df[col] = '-'
    for i in range(1, 4):
        col_name = f'{i}지망'
        if col_name in students_df.columns:
            students_df[f'{i}지망_Key'] = students_df[col_name].apply(parse_preference_key)

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