# ============================================
# 환경 변수 로드 (가장 먼저 실행되어야 함)
# ============================================
from dotenv import load_dotenv
load_dotenv()

# ============================================
# 애플리케이션 수명 주기(lifespan) 관리
# ============================================
from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.common.psycopg_pool_conf import psycopg_pool
from api.common.sqlalchemy_conf import engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.getLogger(__name__).info("애플리케이션 시작")
    # psycopg_pool은 이벤트 루프에서 명시적으로 열어야 함
    await psycopg_pool.open()
    # 앱 시작 ~ 종료까지 대기
    yield
    logging.getLogger(__name__).info("애플리케이션 종료")
    # psycopg_pool 닫기
    await psycopg_pool.close()  # 앱 종료 시 풀 닫기
    # SQLAlchemy 엔진 닫기
    await engine.dispose()

# ============================================
# FastAPI 애플리케이션 인스턴스 생성
# ============================================
# FastAPI() 생성자를 호출하여 애플리케이션 객체를 생성
# - API 문서에 표시될 제목, 설명, 버전을 설정
app = FastAPI(
    title="LangChain과 FastAPI를 활용한 웹 애플리케이션 개발",
    description="LangChain과 FastAPI를 활용하여 웹 애플리케이션을 개발하는 방법을 단계별로 설명",
    version="1.0.0",
    lifespan=lifespan  # 애플리케이션 수명 주기 관리 함수 지정
)

# ============================================
# 로그 설정
# ============================================
# 라우터 임포트 전에 로깅 설정을 먼저 해야 라우터 로드 시 발생하는 로그가 출력됨
import logging
# 루트 로거 가져오기
logger = logging.getLogger()

# 기존 핸들러 제거
# - uvicorn.run(reload=True) 옵션을 사용할 경우 main.py가 두번 실행
# - 따라서 로그 핸들러도 두 번 등록되어 로그 메시지가 중복 출력되는 문제가 발생
# - 기존 핸들러를 제거하여 중복 로그 문제를 해결
logger.handlers.clear()

# 컬러 로그 핸들러 설정: 로그 레벨에 따라 색상을 다르게 표시하여 가독성 향상
import colorlog
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(levelname)s%(reset)s:     '
    '%(cyan)s%(name)s%(reset)s.%(yellow)s%(funcName)s()%(reset)s: '
    '%(green)s%(message)s%(reset)s',
    log_colors={
        'DEBUG': 'white',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
))
logger.addHandler(handler)

# 로그 레벨 설정: INFO 레벨로 설정하여 디버그 메시지는 출력되지 않도록 함
logger.setLevel(logging.INFO)

# ============================================
# CORS(Cross-Origin Resource Sharing) 설정
# ============================================
from fastapi.middleware.cors import CORSMiddleware

# CORS 미들웨어 추가: 모든 출처, 메서드, 헤더 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # 모든 출처 허용 (운영 환경에서는 특정 도메인만 허용 권장)
    allow_credentials=True, # 쿠키, 인증 헤더 등을 포함한 요청 허용
    allow_methods=["*"],    # 모든 HTTP 메서드 허용 (GET, POST, PUT, DELETE 등)
    allow_headers=["*"],    # 모든 헤더 허용
)

# ============================================
# 헬스 체크
# ============================================
@app.get("/")
async def health_check():
    return {"status": "ok"}


# ============================================
# 외부 라우터 추가
# ============================================
# API 기능별로 분리된 라우터 모듈 임포트
from api.auth_service import controller as auth_controller
from api.history_service import controller as chat_history_controller
from api.chat_service import controller as chat_service_controller
from api.upload_service import controller as upload_service_controller
# from api.sec01_dev_environment import controller as sec01_controller
# from api.sec02_text_chat import controller as sec02_controller
# from api.sec03_messages import controller as sec03_controller
# from api.sec04_structured_output import controller as sec04_controller
# from api.sec05_create_agent import controller as sec05_controller
# from api.sec06_chat_history import controller as sec06_controller
# from api.sec07_tool_calling import controller as sec07_controller
# from api.sec08_rag import controller as sec08_controller
# from api.sec09_multi_agent import controller as sec09_controller

# 라우터 등록: FastAPI 애플리케이션에 라우터 추가
app.include_router(auth_controller.router)
app.include_router(chat_history_controller.router)
app.include_router(chat_service_controller.router)
app.include_router(upload_service_controller.router)
# app.include_router(sec01_controller.router)
# app.include_router(sec02_controller.router)
# app.include_router(sec03_controller.router)
# app.include_router(sec04_controller.router)
# app.include_router(sec05_controller.router)
# app.include_router(sec06_controller.router)
# app.include_router(sec07_controller.router)
# app.include_router(sec08_controller.router)
# app.include_router(sec09_controller.router)

# ============================================
# 전역 예외 처리기 등록
# ============================================
from api.common.exception_handler import register_exception_handler
register_exception_handler(app)

# ============================================
# 애플리케이션 시작
# ============================================
import uvicorn

if __name__ == "__main__":
    logger.info("uvicorn(유비콘) 서버로 FastAPI 애플리케이션을 실행")
    uvicorn.run(
        # 유비콘(비동기 서버)가 실행할 애플리케이션
        # reload=False일 경우: app을 제공할 수 있음
        # reload=True 일 경우: "main:app"과 같이 문자열로 제공해야 함
        # main 모듈을 찾아 app을 재시작해야 하므로 모듈 이름이 필요
        "main:app",         
        host="localhost",   # 0.0.0.0은 모든 네트워크 인터페이스에서 접근 가능
        port=80,            # 서버가 실행될 포트 번호     
        reload=True,        # 코드 변경 시 자동으로 서버를 재시작
        access_log=True,    # HTTP 요청/응답 접근 로그 활성화
    )