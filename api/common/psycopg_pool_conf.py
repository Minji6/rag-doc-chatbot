import os
from psycopg_pool import AsyncConnectionPool

###############################################################
# psycopg_pool 설정: 대화 기록을 위한 연결에 사용
###############################################################
psycopg_pool = AsyncConnectionPool(
    os.getenv("DATABASE_URL"), # type: ignore
    min_size=2,
    max_size=5,
    kwargs={"autocommit": True},
    open=False  # FastAPI lifespan에서 명시적으로 열도록 설정
)

