import logging
from typing import Annotated

from fastapi import APIRouter, Form
from fastapi.responses import PlainTextResponse

from api.history_service.agent_history_postgressql import HistoryPostgreSQLAgentDep

# 로거 생성
logger = logging.getLogger(__name__)

# 라우터 생성
router = APIRouter(prefix="/chat_history", tags=["chat_history"])

# -----------------------------------------------------
# Agent를 사용하는 대화 엔드포인트
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

# -----------------------------------------------------
# 대화 ID의 지난 대화 내용 응답을 제공
# -----------------------------------------------------
@router.get("/get-history-postgresql")
async def get_history_postgresql(
    conversation_id: str,
    agent: HistoryPostgreSQLAgentDep
):
    # HistoryPostgreSQLAgent를 사용하여 대화 기록 조회
    return await agent.get_history(conversation_id)