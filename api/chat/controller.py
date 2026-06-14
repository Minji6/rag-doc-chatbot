from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    return ChatResponse(
        answer=f"[연동 성공] 백엔드가 메시지를 수신했습니다: '{request.message}'"
    )
