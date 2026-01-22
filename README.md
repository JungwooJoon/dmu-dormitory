# 기숙사 선발 및 호실 배정 시스템 (Dormitory Management System)

이 시스템은 학생들의 통학 거리와 성적을 기반으로 기숙사 입사 대상자를 선발하고, 선발된 인원들을 학과 및 생활 패턴에 맞춰 최적의 호실로 배정하는 FastAPI 기반 웹 애플리케이션입니다.

---

## 주요 기능

### 1단계: 기숙사 선발 (Selection)

* **거리 계산**: 카카오 로컬 API를 사용하여 주소지 정보를 기반으로 통학 거리를 자동 계산합니다.
* **성적 반영**: 업로드된 성적 데이터를 가중치에 따라 합산합니다.
* **자동 선발**: 설정된 정원에 따라 남/여별 최종 합격자를 선발합니다.
* **결과 정렬**: 성별, 배정된 방 유형, 최종 점수순으로 정렬된 리스트를 제공합니다.

### 2단계: 호실 배정 (Room Assignment)

* **룸메이트 매칭**: 상호 희망하는 룸메이트를 1순위로 우선 배정합니다.
* **최적화 알고리즘**: 학과(학부), 생활 패턴(취침 시간 등) 점수를 계산하여 가장 적합한 쌍을 매칭합니다.
* **유연한 타입 인식**: 엑셀 데이터의 다양한 텍스트 형식을 정규표현식으로 인식하여 방 유형(A~G형)을 자동 매핑합니다.
* **데이터 내보내기**: 최종 배정 결과를 엑셀(.xlsx) 파일로 다운로드할 수 있습니다.

---

## 사전 요구 사항

* **Python 3.12+**
* **카카오 REST API 키**: 거리 계산 기능을 위해 필요합니다.

---

## 설치 및 실행 방법

1. **가상환경 생성 및 활성화**
   ```bash
   python -m venv .venv
   source .venv/Scripts/activate  # Windows
   # source .venv/bin/activate    # Mac/Linux
   ```
2. **필수 라이브러리 설치**
   ```bash
   pip install -r requirements.txt
   ```
3. **서버 실행**
   ```bash
   uvicorn main:app --reload
   ```
## 데이터 양식 가이드
1. 학생 명단 (Excel)
- 필수 컬럼: 학번, 성명, 성별, 주소, 성적, 기숙사 실
- 배정용 컬럼: 학과, 희망룸메이트, 취침시간, 청소주기 등
2. 방 정보 (Excel)
- 필수 컬럼: Room_No, Type, sex, room(정원)
- Type 예시: A형, B형, C형 등

## 프로젝트 파일 구조
   ```Plaintext
   project_root/
   │
   ├── main.py                # FastAPI 웹 서버 및 엔드포인트 관리
   ├── logic_selection.py     # 1단계: 입사생 선발 알고리즘 로직
   ├── logic_assignment.py    # 2단계: 상세 호실 배정 알고리즘 로직
   │
   ├── templates/             # HTML UI 파일 보관
   │   ├── base.html          # 공통 레이아웃
   │   ├── selection.html     # 선발 페이지
   │   └── assignment.html    # 호실 배정 페이지
   │
   ├── README.md              # 프로젝트 안내서
   └── requirements.txt       # 필수 라이브러리 목록
   ```