import pandas as pd
import re
import logging
from itertools import combinations
from typing import Dict, List, Tuple, Optional
from io import BytesIO
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


# [2] 양식 생성 함수
def generate_template_files():
    # 1. 학생 양식 컬럼 구성 (요청하신 순서 그대로)
    student_columns = [
        "학번", "성명", "성별", "1지망", "2지망", "3지망", "학과", "학년",
        "집주소", "성적", "본인 핸드폰 번호", "부모님 핸드폰 번호", "이메일",
        "계좌번호", "은행", "생활패턴", "흡연여부", "희망하는 룸메이트 기재", "우선선발"
    ]

    # 쌤플 데이터 한 줄 추가 (작성 가이드용)
    sample_student_data = [{
        "학번": "20260001", "성명": "홍길동", "성별": "남자",
        "1지망": "1인실(A형)", "2지망": "2인실", "3지망": "4인실",
        "학과": "컴퓨터공학과", "학년": "1", "집주소": "강원특별자치도 강릉시...",
        "성적": "4.5", "우선선발": "X"
    }]

    # 데이터프레임 생성
    df_student = pd.DataFrame(sample_student_data, columns=student_columns)

    # 메모리 버퍼에 엑셀 파일 생성
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_student.to_excel(writer, sheet_name='학생 정보', index=False)
        # (방 정보 템플릿 로직이 있다면 여기에 추가)

    stu_template = output.getvalue()

    return stu_template, None  # (두 번째 인자는 방 정보 템플릿용)


# [3] 최적 룸메이트 쌍 산출
def find_best_pair_info(unassigned_students: pd.DataFrame) -> Optional[Dict]:
    possible_pairs = []
    # 학부 정보가 없는 경우를 대비해 매핑 적용
    if '학부' not in unassigned_students.columns:
        unassigned_students['학부'] = unassigned_students['학과'].map(MAJOR_TO_FACULTY_MAP)

    student_tuples = list(unassigned_students.itertuples(index=True))

    for s1, s2 in combinations(student_tuples, 2):
        score, reasons = 0, []
        # 기본 조건: 흡연 여부 일치
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

            # 추가 조건: 생활 패턴 일치
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


# [4] 메인 배정 로직
async def assign_dorm_rooms(student_df: pd.DataFrame, room_config_df: pd.DataFrame):
    logger.info("호실 배정 알고리즘 실행 (In-Memory)")

    # 타입 키 정규화 함수 (예: 'A형(2인실)' -> 'A형')
    def normalize_type(text):
        if pd.isna(text): return "기타"
        text = str(text).strip()
        match = re.search(r'([A-G]형)', text)
        return match.group(1) if match else text

    # 데이터 정제: 1단계 선발에서 부여된 '배정된 방' 정보가 student_df에 포함되어 있어야 합니다.
    # 만약 컬럼명이 다르다면 해당 시스템의 명칭에 맞춰 수정이 필요합니다.
    target_type_col = '배정된 방' if '배정된 방' in student_df.columns else '1지망'

    student_df['타입_매칭용'] = student_df[target_type_col].apply(normalize_type)
    room_config_df['타입_매칭용'] = room_config_df['유형'].apply(normalize_type)

    # 성별 공백 제거 및 한글 컬럼명 반영
    student_df['성별'] = student_df['성별'].astype(str).str.strip()
    room_config_df['성별'] = room_config_df['성별'].astype(str).str.strip()

    # 가용 방 사전 구축 (한글 컬럼명 반영)
    available_rooms = {}
    for (dtype, sex), group in room_config_df.groupby(['타입_매칭용', '성별']):
        if dtype not in available_rooms: available_rooms[dtype] = {}
        # '호수' 컬럼을 방 번호로 사용
        available_rooms[dtype][sex] = sorted(list(set(group['호수'].astype(str).tolist())))

    final_results = []

    # 타입/성별 그룹별 배정
    for (dorm_type, gender), group in student_df.groupby(['타입_매칭용', '성별']):
        unassigned = group.copy()
        rooms = available_rooms.get(dorm_type, {}).get(gender, []).copy()

        if not rooms:
            for idx in unassigned.index:
                info = unassigned.loc[idx].to_dict()
                info.update({'방 번호': '배정 실패', '선정 이유': f'{dorm_type} 빈 방 없음', 'status': 'failed'})
                final_results.append(info)
            continue

        assigned_set = set()

        # [A] 상호 희망 룸메이트 우선 처리
        for idx, std in unassigned.iterrows():
            if idx in assigned_set: continue

            # 학번 추출 (숫자만)
            match = re.search(r'\d+', str(std.get('희망룸메이트', '')))
            target_id = match.group() if match else None

            if target_id:
                # 상대방도 나를 지목했는지 확인
                target_df = unassigned[
                    (unassigned['학번'].astype(str) == target_id) & (~unassigned.index.isin(assigned_set))]
                if not target_df.empty:
                    target_std = target_df.iloc[0]
                    if str(std['학번']) in str(getattr(target_std, '희망룸메이트', '')) and rooms:
                        room_num = rooms.pop(0)
                        for r_idx in [idx, target_df.index[0]]:
                            info = unassigned.loc[r_idx].to_dict()
                            info.update({'방 번호': room_num, '선정 이유': '상호 희망', 'status': 'assigned'})
                            final_results.append(info)
                            assigned_set.add(r_idx)

        unassigned.drop(index=list(assigned_set), inplace=True)

        # [B] 알고리즘 기반 매칭 및 잔여 처리
        while not unassigned.empty:
            # 2명 이상 남아있고 방이 있는 경우 최적의 쌍 매칭
            if len(unassigned) >= 2 and rooms:
                best = find_best_pair_info(unassigned)
                if best:
                    room_num = rooms.pop(0)
                    for r_idx in best['pair']:
                        info = unassigned.loc[r_idx].to_dict()
                        info.update({'방 번호': room_num, '선정 이유': best['reason'], 'status': 'assigned'})
                        final_results.append(info)
                    unassigned.drop(index=list(best['pair']), inplace=True)
                    continue

            # 혼자 남았거나 쌍을 찾지 못한 경우 일반 배정
            if not unassigned.empty and rooms:
                idx = unassigned.index[0]
                room_num = rooms.pop(0)
                info = unassigned.loc[idx].to_dict()
                info.update({'방 번호': room_num, '선정 이유': '일반 배정', 'status': 'assigned'})
                final_results.append(info)
                unassigned.drop(index=[idx], inplace=True)
            else:
                # 방이 부족하여 배정되지 못한 인원 처리
                for idx in unassigned.index:
                    info = unassigned.loc[idx].to_dict()
                    info.update({'방 번호': '배정 보류', '선정 이유': '방 부족', 'status': 'waitlist'})
                    final_results.append(info)
                break

    return final_results