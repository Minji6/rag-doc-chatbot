import uuid
import logging
from fastapi import APIRouter
from pydantic import BaseModel

from api.chat_service.langgraph.supervisor import ChatbotSupervisor
from api.history_service.agent_history_inmemory import HistoryInMemoryAgent

router = APIRouter(prefix="/api/chat", tags=["chat"])
logger = logging.getLogger(__name__)

_supervisor = ChatbotSupervisor()
_history_agent = HistoryInMemoryAgent()


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    conversation_id = request.conversation_id or str(uuid.uuid4())

    history = await _history_agent.get_history(conversation_id)
    logger.info(f"[{conversation_id}] 이전 대화 {len(history['messages'])}개 로드")

    result = await _supervisor.run(
        user_inquiry=request.message,
        user_role="guest",
        messages=history["messages"],
    )
    answer = result["message"]

    await _history_agent.save_exchange(request.message, answer, conversation_id)
    logger.info(f"[{conversation_id}] 대화 저장 완료")

    return ChatResponse(answer=answer, conversation_id=conversation_id)