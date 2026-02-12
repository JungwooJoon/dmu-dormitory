import pandas as pd
import re
import logging
from itertools import combinations
from typing import Dict, List, Tuple, Optional
import io

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DmuAssignment")

# [1] 데이터 매핑 정의
MAJOR_TO_FACULTY_MAP = {
    '기계공학과': '기계공학부', '기계설계공학과': '기계공학부', '자동화공학과': '로봇자동화공학부', '로봇소프트웨어과': '로봇자동화공학부',
    '전기공학과': '전기전자통신공학부', '반도체전자공학과': '전기전자통신공학부', '정보통신공학과': '전기전자통신공학부', '소방안전관리과': '전기전자통신공학부',
    '웹응용소프트웨어공학과': '컴퓨터공학부', '컴퓨터소프트웨어공학과': '컴퓨터공학부', '인공지능소프트웨어학과': '컴퓨터공학부', '생명화학공학과': '생활환경공학부',
    '바이오융합공학과': '생활환경공학부', '건축과': '생활환경공학부', '실내건축디자인과': '생활환경공학부', '시각디자인과': '생활환경공학부',
    'AR·VR콘텐츠디자인과': '생활환경공학부', '경영학과': '경영학부', '세무회계학과': '경영학부', '유통마케팅학과': '경영학부',
    '호텔관광학과': '경영학부', '경영정보학과': '경영학부', '빅데이터경영과': '경영학부', '자유전공학과': '자유전공학부'
}


# [2] 최적 룸메이트 쌍 산출 (2인 단위 알고리즘 매칭용)
def find_best_pair_info(unassigned_students: pd.DataFrame) -> Optional[Dict]:
    possible_pairs = []
    if '학부' not in unassigned_students.columns:
        unassigned_students['학부'] = unassigned_students['학과'].map(MAJOR_TO_FACULTY_MAP)

    student_tuples = list(unassigned_students.itertuples(index=True))

    for s1, s2 in combinations(student_tuples, 2):
        score, reasons = 0, []
        if getattr(s1, '흡연여부', '') == getattr(s2, '흡연여부', ''):
            if s1.학과 == s2.학과:
                score = 10
                reasons = ['흡연 동일', '동일 학과']
            elif getattr(s1, '학부', '') == getattr(s2, '학부', ''):
                score = 8
                reasons = ['흡연 동일', '동일 학부']
            else:
                score = 6
                reasons = ['흡연 동일']

            if getattr(s1, '생활패턴', '') == getattr(s2, '생활패턴', ''):
                score += 2
                reasons.append('패턴 동일')

        if score > 0:
            possible_pairs.append({'pair': (s1.Index, s2.Index), 'score': score, 'reason': ', '.join(reasons)})

    if not possible_pairs:
        if len(unassigned_students) >= 2:
            return {'pair': (unassigned_students.index[0], unassigned_students.index[1]), 'reason': '랜덤 배정'}
        return None

    return sorted(possible_pairs, key=lambda x: x['score'], reverse=True)[0]


# [3] 메인 배정 로직
async def assign_dorm_rooms(student_df: pd.DataFrame, room_config_df: pd.DataFrame):
    logger.info("호실 배정 알고리즘 실행 (In-Memory)")

    # 타입 키 정규화 함수
    def normalize_type(text):
        if pd.isna(text): return "기타"
        text = str(text).strip()
        match = re.search(r'([A-G]형)', text)
        return match.group(1) if match else text

    target_type_col = '배정된 방' if '배정된 방' in student_df.columns else '1지망'
    student_df['타입_매칭용'] = student_df[target_type_col].apply(normalize_type)
    room_config_df['타입_매칭용'] = room_config_df['유형'].apply(normalize_type)
    student_df['학번'] = student_df['학번'].astype(str).str.replace(r'\s+', '', regex=True)

    # [핵심] room_capacities 변수 생성
    # 기숙사 설정 파일에서 각 유형별 인실(capacity) 정보를 딕셔너리로 저장
    # 예: {'A형': 2, 'E형': 4}
    room_capacities = {}
    if '인실' in room_config_df.columns:
        # 인실 컬럼에서 숫자만 추출하여 매핑
        for _, row in room_config_df.iterrows():
            ctype = row['타입_매칭용']
            cap_val = re.search(r'\d', str(row['인실']))
            if cap_val:
                room_capacities[ctype] = int(cap_val.group())

    # 기본값 설정 (정보가 없을 경우 2인실로 간주)
    for t in student_df['타입_매칭용'].unique():
        if t not in room_capacities: room_capacities[t] = 2

    # 성별 공백 제거
    student_df['성별'] = student_df['성별'].astype(str).str.strip()
    room_config_df['성별'] = room_config_df['성별'].astype(str).str.strip()

    # 가용 방 사전 구축
    available_rooms = {}
    for (dtype, sex), group in room_config_df.groupby(['타입_매칭용', '성별']):
        if dtype not in available_rooms: available_rooms[dtype] = {}
        available_rooms[dtype][sex] = sorted(list(set(group['호수'].astype(str).tolist())))

    final_results = []
    assigned_set = set()

    # 타입/성별 그룹별 배정
    for (dorm_type, gender), group in student_df.groupby(['타입_매칭용', '성별']):
        unassigned = group.copy()
        rooms = available_rooms.get(dorm_type, {}).get(gender, []).copy()

        # 현재 방의 정원 확인
        current_capacity = room_capacities.get(dorm_type, 2)

        if not rooms:
            for idx in unassigned.index:
                if idx in assigned_set: continue
                info = unassigned.loc[idx].to_dict()
                info.update({'방 번호': '배정 실패', '선정 이유': f'{dorm_type} 빈 방 없음', 'status': 'failed'})
                final_results.append(info)
                assigned_set.add(idx)
            continue

        current_indices = list(unassigned.index)
        for idx in current_indices:
            if idx in assigned_set: continue

            std = unassigned.loc[idx]
            # [수정] 내 학번을 문자열로 바꾸고 모든 공백 제거
            my_id = re.sub(r'\s+', '', str(std['학번']))
            pref_text = str(std.get('희망룸메이트', ''))

            # 현재 방 정원에 따른 텍스트 섹션 분리
            parts = re.split(r'4인(?:호|실)?', pref_text, flags=re.IGNORECASE)
            target_section = parts[-1] if current_capacity == 4 and len(parts) > 1 else parts[0]

            # 섹션 내에서 5자리 이상 학번 추출 및 정문화
            target_ids = [re.sub(r'\s+', '', tid) for tid in re.findall(r'\d{5,}', target_section)]

            matched_indices = []
            for tid in target_ids:
                if tid == my_id: continue

                # [핵심] 비교 대상 학번도 실시간으로 문자열 정규화하여 검색
                candidate_df = unassigned[
                    (unassigned['학번'].astype(str).str.replace(r'\s+', '', regex=True) == tid) &
                    (~unassigned.index.isin(assigned_set))
                    ]

                if not candidate_df.empty:
                    c_idx = candidate_df.index[0]
                    c_std = candidate_df.iloc[0]
                    c_pref = str(c_std.get('희망룸메이트', ''))

                    # 상대방 글 속의 모든 학번을 추출하여 내 학번이 있는지 확인
                    other_ids = [re.sub(r'\s+', '', oid) for oid in re.findall(r'\d{5,}', c_pref)]

                    if my_id in other_ids:
                        matched_indices.append(c_idx)

                if len(matched_indices) >= current_capacity - 1:
                    break

            if matched_indices and rooms:
                room_num = rooms.pop(0)
                all_roomies = [idx] + matched_indices
                for r_idx in all_roomies:
                    info = unassigned.loc[r_idx].to_dict()
                    info.update({
                        '방 번호': room_num,
                        '선정 이유': f'상호 희망 ({current_capacity}인)',
                        'status': 'assigned'
                    })
                    final_results.append(info)
                    assigned_set.add(r_idx)

        # 배정된 인원 데이터프레임에서 제거
        unassigned = unassigned.drop(index=list(assigned_set), errors='ignore')

        # [B] 알고리즘 기반 매칭 및 잔여 처리 (순차 배정)
        while not unassigned.empty:
            # 2인 단위 최적 쌍 찾기
            if len(unassigned) >= 2 and rooms:
                best = find_best_pair_info(unassigned)
                if best:
                    room_num = rooms.pop(0)
                    for r_idx in best['pair']:
                        info = unassigned.loc[r_idx].to_dict()
                        info.update({'방 번호': room_num, '선정 이유': best['reason'], 'status': 'assigned'})
                        final_results.append(info)
                        assigned_set.add(r_idx)
                    unassigned = unassigned.drop(index=list(best['pair']))
                    continue

            # 혼자 남았거나 방이 하나뿐인 경우 일반 배정
            if not unassigned.empty and rooms:
                idx = unassigned.index[0]
                room_num = rooms.pop(0)
                info = unassigned.loc[idx].to_dict()
                info.update({'방 번호': room_num, '선정 이유': '일반 배정', 'status': 'assigned'})
                final_results.append(info)
                assigned_set.add(idx)
                unassigned = unassigned.drop(index=[idx])
            else:
                # 방 부족 시 대기 처리
                for idx in unassigned.index:
                    info = unassigned.loc[idx].to_dict()
                    info.update({'방 번호': '배정 보류', '선정 이유': '방 부족', 'status': 'waitlist'})
                    final_results.append(info)
                    assigned_set.add(idx)
                break

    return final_results
