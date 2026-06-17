from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from api.chat.supervisor import YouthPolicySupervisorDep

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    conversation_id: str = "default"
    user_id: str | None = None


class ChatResponse(BaseModel):
    answer: str


@router.post("")
async def chat(request: ChatRequest, supervisor: YouthPolicySupervisorDep):
    async def generate():
        async for chunk in supervisor.stream(
            inquiry=request.message,
            conversation_id=request.conversation_id,
            user_id=request.user_id,
        ):
            yield chunk

    return StreamingResponse(generate(), media_type="text/plain")
