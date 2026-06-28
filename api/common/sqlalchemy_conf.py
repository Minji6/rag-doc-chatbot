import os
from typing import Annotated, AsyncGenerator
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine, AsyncAttrs
from sqlalchemy.orm import DeclarativeBase

###############################################################
# SQLAlchemy 비동기 엔진(커넥션풀) 생성
###############################################################
# - pool_size: 기본 커넥션 풀 크기
# - max_overflow: 피크 시 추가 커넥션 수, 최대 커넥션 수 = pool_size + max_overflow
# - pool_pre_ping: 유휴 커넥션 체크. 끊어진 커넥션을 재사용하는 문제를 방지
# - echo=True 옵션은 SQLAlchemy가 실행하는 모든 SQL 문을 콘솔에 출력하도록 설정
engine = create_async_engine(
    os.getenv("DATABASE_URL_SQLALCHEMY"), # type: ignore
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=300,    # 5분마다 커넥션 교체 - DB auto-suspend 주기와 맞춤
    pool_timeout=10,     # 커넥션 획득 대기 최대 10초
    echo=False
)

###############################################################
# ORM 작업 세션 팩토리 생성
###############################################################
# - autoflush: 파이썬 메모리(세션)에 쌓여 있는 변경사항을 DB와 자동 동기화(flush)할지 여부
#   - False로 설정: 명시적으로 flush()를 호출해야 DB에 반영됨
#   - flush는 파이썬 메모리(세션) ↔ DB의 차이를 맞추는 작업, 커밋(commit)은 아님
# - autocommit: 각 SQL문 실행후 자동 커밋 여부 (기본값: False)
SessionMaker = async_sessionmaker(
    bind=engine, 
    autoflush=True, 
    autocommit=False
)

###############################################################
# ORM 작업 세션(AsyncSession)을 제공하는 제너레이터 생성 함수
###############################################################
# - Depends로 주입하여 사용, 한 요청에 여러 번 Depends해도 같은 세션 객체 사용
# - 트랜잭션 처리 (commit, rollback) 포함
# - AsyncGenerator[AsyncSession, None]:
#   - 비동기 세션 객체를 제공하는 비동기 제너레이터
#   - 외부처리결과를 받을 수는 있지만, 리턴값은 없으므로 타입 힌트는 두개만 존재
#   - 외부처리가 끝나면 yield 다음의 코드가 실행되어 세션을 정리함
#   - 제너레이터가 종료될 때 finally 블록이 실행되어 세션이 닫힘
#   - None: AsyncSession 객체만 생성하여 제공하고, 외부처리결과는 없음
async def get_orm_session() -> AsyncGenerator[AsyncSession, None]:
    # 세션 생성
    orm_session = SessionMaker()
    try:
        # AsyncSession 반환하고 get_session 함수는 일시 중단 상태가 됨
        yield orm_session
        # 컨트롤러 함수가 정상 종료되면 yield 다음 코드부터 실행
        # 트랜잭션 커밋
        await orm_session.commit()
    except:
        # 컨트롤러 함수에서 예외 발생 시 롤백 수행
        await orm_session.rollback()
        # 동일한 예외를 다시 발생시켜 상위 호출 스택으로 전달
        raise
    finally:
        # 제너레이터가 종료될 때 세션을 닫음
        await orm_session.close()

###############################################################
# 의존성 타입 별칭 정의
###############################################################
OrmSessionDep = Annotated[AsyncSession, Depends(get_orm_session)]


###############################################################
# 엔티티 클래스의 부모 클래스 정의
###############################################################
# - AsyncAttrs와 DeclarativeBase를 상속
# - AsyncAttrs: SQLAlchemy의 비동기 ORM 기능을 제공하는 클래스
# - DeclarativeBase: SQLAlchemy의 선언적 매핑 기능을 제공하는 클래스
class Base(AsyncAttrs, DeclarativeBase):
    pass