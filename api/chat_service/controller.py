import logging
<<<<<<< Updated upstream
from typing import Annotated
from fastapi import APIRouter, Form, HTTPException
=======
from typing import Annotated, Literal
from fastapi import APIRouter, Form
>>>>>>> Stashed changes
from fastapi.responses import JSONResponse
from api.chat_service.langgraph.supervisor import ChatbotSupervisor

from api.chat_service.langgraph.supervisor import ChatbotSupervisorDep
from api.chat_service.langgraph.constants import ROLE_USER, ROLE_GUEST
from api.auth_service.service import UserServiceDep
from api.common.sqlalchemy_conf import OrmSessionDep

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])

supervisor = ChatbotSupervisor() 


@router.post("/service", response_class=JSONResponse)
async def chat(
    message: Annotated[str, Form()],
    conversation_id: Annotated[str, Form()],
    user_id: Annotated[str, Form()] = "guest",  # 기본값 설정
):
    logger.info(f"[{conversation_id}] 메시지 수신: {message[:30]}...") # 기본값 설정

    result = await supervisor.run(
        inquiry=message,
        user_id=user_id,
    )

    return JSONResponse(content={
        "conversation_id": conversation_id,
        "response": result,
    })
