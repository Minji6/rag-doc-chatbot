import logging
from typing import Annotated
from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/service", response_class=JSONResponse)
async def chat(
    message: Annotated[str, Form()],
    conversation_id: Annotated[str, Form()],
):
    logger.info(f"[{conversation_id}] 메시지 수신: {message[:30]}...")
    return JSONResponse(content={
        "conversation_id": conversation_id,
        "message": message  
    })