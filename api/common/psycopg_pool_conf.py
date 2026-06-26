import os
from psycopg_pool import AsyncConnectionPool

###############################################################
# psycopg_pool 설정: 대화 기록을 위한 연결에 사용
###############################################################
psycopg_pool = AsyncConnectionPool(
    os.getenv("DATABASE_URL"), # type: ignore
    min_size=1,
    max_size=2,
    max_idle=300,           # 5분 이상 유휴 연결은 자동 회수 후 재생성
    reconnect_timeout=30,   # 재연결 실패 시 30초까지 재시도
    kwargs={"autocommit": True},
    open=False,             # FastAPI lifespan에서 명시적으로 열도록 설정
    check=AsyncConnectionPool.check_connection,  # 연결 반환 전 유효성 검사
)

