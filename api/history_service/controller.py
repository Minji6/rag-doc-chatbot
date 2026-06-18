import logging
from typing import Annotated

from fastapi import APIRouter, Form, Query
from fastapi.responses import JSONResponse, PlainTextResponse

from api.history_service.agent_history_inmemory import HistoryInMemoryAgentDep
from api.history_service.agent_history_postgresql import HistoryPostgreSQLAgentDep

# 로거 생성
logger = logging.getLogger(__name__)

# 라우터 생성
router = APIRouter(prefix="/chat_history", tags=["chat_history"])

# -----------------------------------------------------
# HistoryPostgreSQLAgent: 대화 기억을 유지하면서 텍스트 대화
# -----------------------------------------------------
@router.post("/chat-history-postgresql", response_class=PlainTextResponse)
async def chat_history_postgresql(
    message: Annotated[str, Form()],
    conversation_id: Annotated[str, Form()],
    agent: HistoryPostgreSQLAgentDep
):
    # 스트리밍 대화 실행
    response = await agent.run(message, conversation_id)
    return response

# 대화 ID의 지난 대화 내용 응답을 제공
@router.get("/get-history-postgresql")
async def get_history_postgresql(
    conversation_id: Annotated[str, Query(description="조회할 대화 ID")],
    agent: HistoryPostgreSQLAgentDep
):
    # HistoryPostgreSQLAgent를 사용하여 대화 기록 조회
    return await agent.get_history(conversation_id)

# 대화 ID의 지난 대화 내용을 모두 삭제
@router.delete("/clear-history-postgresql")
async def clear_history_postgresql(
    conversation_id: Annotated[str, Query(description="삭제할 대화 ID")],
    agent: HistoryPostgreSQLAgentDep
):
    # HistoryPostgreSQLAgent를 사용하여 대화 기록 삭제
    return await agent.clear_history(conversation_id)

# -----------------------------------------------------
# InMemoryAgent: 대화 기억을 유지 하지 않음, guest
# -----------------------------------------------------
@router.post("/chat-history-inmemory",
             response_class=PlainTextResponse)

async def chat_history_inmemory(
    message: Annotated[str, Form()],
    conversation_id: Annotated[str, Form()],
    agent: HistoryInMemoryAgentDep
):
    response = await agent.run(message, conversation_id)
    return response

# 대화 ID의 지난 대화 내용 응답을 제공
@router.get("/get-history-inmemory", response_class=JSONResponse)
async def get_history_inmemory(
    conversation_id: Annotated[str, Query()],
    agent: HistoryInMemoryAgentDep
):
    history = await agent.get_history(conversation_id)
    return history

# 대화 ID의 지난 대화 내용 응답을 삭제
@router.delete("/clear-history-inmemory", response_class=JSONResponse)
async def clear_history_inmemory(
    conversation_id: Annotated[str, Query(description="삭제할 대화 ID")],
    agent: HistoryInMemoryAgentDep
):
    return await agent.clear_history(conversation_id)
