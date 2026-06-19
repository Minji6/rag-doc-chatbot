import logging
from typing import Annotated, Literal

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from api.history_service.agent_dependency import HistoryAgentByQueryDep, build_thread_id


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
    # 응답의 식별자를 원래 conversation_id로 되돌림 (합성 키 비노출)
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
    # 응답의 식별자를 원래 conversation_id로 되돌림 (합성 키 비노출)
    result["conversation_id"] = conversation_id
    return result
