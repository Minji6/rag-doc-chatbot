import logging
from typing import Annotated
from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse

from api.chat_service.langgraph.supervisor import ChatbotSupervisorDep
from api.chat_service.langgraph.constants import ROLE_USER, ROLE_GUEST
from api.auth_service.service import UserServiceDep
from api.common.sqlalchemy_conf import OrmSessionDep

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/service", response_class=JSONResponse)
async def chat(
    message: Annotated[str, Form()],
    conversation_id: Annotated[str, Form()],
    role: Annotated[str, Form()],
    supervisor: ChatbotSupervisorDep,
    user_service: UserServiceDep,
    session: OrmSessionDep,
    user_id: Annotated[int | None, Form()] = None,
):
    if role not in (ROLE_USER, ROLE_GUEST):
        raise HTTPException(status_code=422, detail=f"role은 '{ROLE_USER}' 또는 '{ROLE_GUEST}'여야 합니다.")

    logger.info(f"[{conversation_id}] 메시지 수신: {message[:30]}...")

    user_profile = None
    if role == ROLE_USER:
        if not user_id:
            raise HTTPException(status_code=422, detail="user role에는 user_id가 필요합니다.")
        user_profile = await user_service.get_user_profile(user_id, session)
        if not user_profile:
            raise HTTPException(status_code=404, detail=f"user_id={user_id} 유저를 찾을 수 없습니다.")

    response = await supervisor.run(user_inquiry=message, user_role=role, user_profile=user_profile)
    return JSONResponse(content={
        "conversation_id": conversation_id,
        "message": response,
    })
