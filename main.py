from fastapi import FastAPI, Request, UploadFile, File, APIRouter
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, Response, StreamingResponse, RedirectResponse, HTMLResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from urllib.parse import quote
import logging
import re
import pandas as pd
import numpy as np
import io
import logic_selection, logic_assignment

app = FastAPI()
templates = Jinja2Templates(directory="templates")
assignment_router = APIRouter(prefix="/assignment")
app.mount("/static", StaticFiles(directory="static"), name="static")

origins = [
    "http://localhost",
    "http://0.0.0.0",
    "http://0.0.0.0:8000",
    "http://localhost:3000",  # 리액트 등 프론트엔드 포트
    "http://127.0.0.1:8000",
    "https://campuslife.dongyang.ac.kr",
    "http://campuslife.dongyang.ac.kr"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 모든 도메인에서의 접속을 허용 (실무에서는 특정 도메인만 지정 권장)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드(GET, POST, PUT, DELETE 등) 허용
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)

# MongoDB를 대체하는 메모리 저장소
storage = {
    "students": pd.DataFrame(),
    "rooms": pd.DataFrame(),
    "app_config": {},
    "selection_result": None,
    "assignment_result": None
}


@app.get("/")
async def root_redirect():
    return RedirectResponse(url="/assignment")


# ==========================================
# 1. 페이지 렌더링 (기본 경로)
# ==========================================
@assignment_router.get("/")
async def read_selection(request: Request):
    return templates.TemplateResponse("selection.html", {"request": request, "page": "selection"})


@assignment_router.get("/room-assignment")
async def read_assignment(request: Request):
    return templates.TemplateResponse("assignment.html", {"request": request, "page": "assignment"})


@assignment_router.get("/export-format")
async def read_export_page(request: Request):
    return templates.TemplateResponse("export.html", {"request": request, "page": "export"})


# ==========================================
# 2. 1단계: 기숙사 선발 관련 기능 (Selection)
# ==========================================
@assignment_router.post("/upload-selection-data")
async def upload_selection(
        students: UploadFile = File(...),
        rooms: UploadFile = File(...),
        config: UploadFile = File(...)
):
    try:
        storage["students"] = pd.read_excel(io.BytesIO(await students.read()), dtype={'본인 핸드폰 번호': str})
        storage["rooms"] = pd.read_excel(io.BytesIO(await rooms.read()))
        c_df = pd.read_excel(io.BytesIO(await config.read()))
        storage["app_config"] = dict(zip(c_df['항목'].astype(str).str.strip(), c_df['값']))
        return {"status": "success", "message": "선발용 데이터와 설정이 로드되었습니다."}
    except Exception as e:
        return {"status": "error", "message": f"업로드 실패: {str(e)}"}


@assignment_router.post("/run-selection")
async def run_selection():
    try:
        if storage["students"].empty or storage["rooms"].empty:
            return {"status": "error", "message": "데이터가 부족합니다. 업로드를 먼저 진행하세요."}

        result_df, _ = await logic_selection.run_selection_process(
            storage["students"].copy(),
            storage["rooms"].copy(),
            storage["app_config"]
        )

        result_df = result_df.replace([np.inf, -np.inf], np.nan)
        result_df = result_df.sort_values(by=['성별', '배정된 방', '최종 점수'], ascending=[True, True, False])
        storage["selection_result"] = result_df.copy()

        fill_values = {
            '최종 점수': 0,
            '통학 점수(70점)': 0,
            '성적 점수(30점)': 0,
            '소요시간(분)': 0,
            '교통수단': '-',
            '배정된 방': '-',
            '배정방식': '-'
        }
        display_df = result_df.fillna(value=fill_values).fillna("-")

        return jsonable_encoder({
            "status": "success",
            "data": display_df.to_dict('records')
        })
    except Exception as e:
        logging.error(f"선발 오류: {str(e)}")
        return {"status": "error", "message": f"서버 오류: {str(e)}"}


@assignment_router.get("/download-results")
async def download_results():
    if storage["selection_result"] is None:
        return {"status": "error", "message": "다운로드할 결과가 없습니다."}

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        storage["selection_result"].to_excel(writer, index=False, sheet_name='선발결과')
    output.seek(0)
    filename = "선발 결과.xlsx"
    encoded_filename = quote(filename)
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}"}
    )


# ==========================================
# 3. 2단계: 호실 배정 관련 기능 (Assignment)
# ==========================================
@assignment_router.post("/upload-assignment-data")
async def upload_assignment_data(students: UploadFile = File(...), rooms: UploadFile = File(...)):
    try:
        storage["students"] = pd.read_excel(io.BytesIO(await students.read()), dtype={'본인 핸드폰 번호': str})
        storage["rooms"] = pd.read_excel(io.BytesIO(await rooms.read()))
        return {"status": "success", "message": "배정용 학생 및 방 정보가 업데이트되었습니다."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@assignment_router.post("/run-room-matching")
async def run_room_matching():
    try:
        if storage["students"].empty or storage["rooms"].empty:
            return {"status": "error", "message": "배정할 데이터가 없습니다."}

        result_list = await logic_assignment.assign_dorm_rooms(
            storage["students"].copy(),
            storage["rooms"].copy()
        )

        df_final = pd.DataFrame(result_list)
        storage["assignment_result"] = df_final.copy()
        final_data = df_final.fillna("-").replace([np.inf, -np.inf], 0).to_dict('records')

        return {"status": "success", "data": final_data}
    except Exception as e:
        logging.error(f"호실 배정 에러: {str(e)}")
        return {"status": "error", "message": str(e)}


@assignment_router.get("/download-assignment-results")
async def download_assignment_results():
    if storage["assignment_result"] is None or storage["assignment_result"].empty:
        return {"status": "error", "message": "결과가 없습니다."}

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        storage["assignment_result"].to_excel(writer, index=False, sheet_name='호실배정결과')
    output.seek(0)
    filename = "호수 배정 결과.xlsx"
    encoded_filename = quote(filename)
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}"}
    )


# ==========================================
# 4. 템플릿 다운로드 기능
# ==========================================
@assignment_router.get("/download-student-template")
async def download_student_template():
    stu_template, _ = logic_assignment.generate_template_files()
    filename = "학생 정보(양식).xlsx"
    encoded_filename = quote(filename)
    return Response(content=stu_template,
                    media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}"})


@assignment_router.get("/download-room-template")
async def download_room_template():
    _, room_template = logic_assignment.generate_template_files()
    filename = "기숙사 방 정보(양식).xlsx"
    encoded_filename = quote(filename)
    return Response(content=room_template,
                    media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}"})


@assignment_router.get("/download-setup-template")
async def download_setup_template():
    # [1] 이미지(image_a20820.png)와 동일한 구성 데이터 생성
    setup_data = [
        {"항목": "카카오키", "값": "", "설명 (참고용)": "카카오 REST API 키"},
        {"항목": "오디세이키", "값": "", "설명 (참고용)": "ODsay API 키"},
        {"항목": "항공페널티", "값": "5", "설명 (참고용)": "항공 수단 가중치"},
        {"항목": "고속철도페널티", "값": "2", "설명 (참고용)": "KTX 및 SRT 수단 가중치"},
        {"항목": "고속버스페널티", "값": "2", "설명 (참고용)": "고속 및 시외버스 수단 가중치"},
        {"항목": "학교주소", "값": "서울시 구로구 경인로 445", "설명 (참고용)": "거리 계산의 도착 지점"},
        {"항목": "재학생성적", "값": "4.5:30,4.0:25,3.5:20,3.0:15,2.5:10", "설명 (참고용)": "학점 구간별 점수 매핑"},
        {"항목": "신입생성적", "값": "950:30,900:25,850:20,800:15,750:10,700:5", "설명 (참고용)": "1000점 만점 구간별 점수 매핑"}
    ]

    df_setup = pd.DataFrame(setup_data)

    output = io.BytesIO()
    # [2] xlsxwriter를 사용하여 시트 생성 및 서식(선택사항) 적용
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_setup.to_excel(writer, sheet_name='설정', index=False)

    excel_data = output.getvalue()
    filename = "설정(양식).xlsx"
    encoded_filename = quote(filename)

    return Response(
        content=excel_data,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}",
            "Access-Control-Expose-Headers": "Content-Disposition",
            "Cache-Control": "no-cache"
        }
    )


@assignment_router.post("/convert-to-official")
async def convert_to_official(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_excel(io.BytesIO(content))

    def format_phone(val):
        if pd.isna(val) or str(val).strip() == "":
            return ""
        s = str(val).split('.')[0].strip()
        s = "".join(filter(str.isdigit, s))
        if len(s) == 10 and s.startswith('1'):
            return '0' + s
        return s

    def get_floor(room_no):
        nums = re.findall(r'\d+', str(room_no))
        if nums:
            return nums[0][0] if len(nums[0]) >= 3 else "1"
        return "1"

    final_df = pd.DataFrame()
    final_df['No'] = range(1, len(df) + 1)
    final_df['Floor'] = df['방 번호'].apply(get_floor) if '방 번호' in df.columns else ""
    final_df['Room_No'] = df['방 번호'] if '방 번호' in df.columns else ""
    final_df['Room_Type'] = df['타입_매칭용'] if '타입_매칭용' in df.columns else ""
    final_df['Hakbun'] = df['학번'].astype(str) if '학번' in df.columns else ""
    final_df['Name'] = df['성명'] if '성명' in df.columns else ""
    final_df['Sex'] = df['성별'] if '성별' in df.columns else ""
    final_df['Dep'] = df['학과'] if '학과' in df.columns else ""
    final_df['Grade'] = df['성적'] if '성적' in df.columns else ""

    if '본인 핸드폰 번호' in df.columns:
        final_df['Phone'] = df['본인 핸드폰 번호'].apply(format_phone)
    else:
        final_df['Phone'] = ""

    final_df['E-mail'] = ""
    final_df['Guardian_Phone'] = ""

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        final_df.to_excel(writer, index=False, sheet_name='Sheet1')

        workbook = writer.book
        worksheet = writer.sheets['Sheet1']

        format_text = workbook.add_format({'num_format': '@'})

        worksheet.set_column('E:E', None, format_text)
        worksheet.set_column('J:J', None, format_text)

    output.seek(0)
    filename = "최종_배정명단.xlsx"
    encoded_filename = quote(filename)

    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}"}
    )


app.include_router(assignment_router)
