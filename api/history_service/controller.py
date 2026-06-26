import logging
from typing import Annotated, Literal

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from api.history_service.agent_dependency import HistoryAgentByQueryDep, build_thread_id
from api.auth_service.service import UserServiceDep
from api.common.sqlalchemy_conf import OrmSessionDep


# 로거 생성
logger = logging.getLogger(__name__)

# 라우터 생성
router = APIRouter(prefix="/chat_history", tags=["chat_history"])

# -----------------------------------------------------
# 대화 기록 조회: role에 따라 PostgreSQL 또는 InMemory 에이전트로 분기
# -----------------------------------------------------
@router.get("/get-history", response_class=JSONResponse)
async def get_history(
    conversation_id: Annotated[str, Query(description="조회할 대화 ID")],
    role: Annotated[Literal["user", "guest"], Query()],
    agent: HistoryAgentByQueryDep,
    user_id: Annotated[str | None, Query()] = None,
):
    thread_id = build_thread_id(role, conversation_id, user_id)
    result = await agent.get_history(thread_id)
    result["conversation_id"] = conversation_id
    return result

# -----------------------------------------------------
# 대화 기록 삭제: role에 따라 PostgreSQL 또는 InMemory 에이전트로 분기
# -----------------------------------------------------
@router.delete("/clear-history", response_class=JSONResponse)
async def clear_history(
    conversation_id: Annotated[str, Query(description="삭제할 대화 ID")],
    role: Annotated[Literal["user", "guest"], Query()],
    agent: HistoryAgentByQueryDep,
    user_id: Annotated[str | None, Query()] = None,
):
    thread_id = build_thread_id(role, conversation_id, user_id)
    result = await agent.clear_history(thread_id)
    result["conversation_id"] = conversation_id
    return result

# -----------------------------------------------------
# 유저별 대화 목록 조회 (user 전용)
# TODO: user_id 파라미터가 실제 요청자 본인인지 검증하는 인증 미들웨어 필요
# -----------------------------------------------------
@router.get("/conversations", response_class=JSONResponse)
async def get_conversations(
    user_id: Annotated[str, Query(description="유저 ID")],
    service: UserServiceDep,
    session: OrmSessionDep,
):
    return await service.get_conversations(user_id, session)
