import logging
from typing import Annotated, Literal
from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import select

from api.chat_service.langgraph.supervisor import ChatbotSupervisorDep
from api.common.sqlalchemy_conf import OrmSessionDep
from api.models.user import User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/service", response_class=JSONResponse)
async def chat(
    message: Annotated[str, Form()],
    conversation_id: Annotated[str, Form()],
    role: Annotated[Literal["user", "guest"], Form()],
    supervisor: ChatbotSupervisorDep,
    session: OrmSessionDep,
    user_id: Annotated[int | None, Form()] = None,
):
    logger.info(f"[{conversation_id}] 메시지 수신: {message[:30]}...")

    user_profile = {}
    if role == "user":
        if not user_id:
            raise HTTPException(status_code=422, detail="user role에는 user_id가 필요합니다.")
        result = await session.execute(select(User).where(User.user_id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=404, detail=f"user_id={user_id} 유저를 찾을 수 없습니다.")
        user_profile = {
            "user_id":               user.user_id,
            "nickname":              user.nickname,
            "birth_date":            str(user.birth_date),
            "region_code":           user.region_code,
            "interested_categories": user.interested_categories,
            "education_level_code":  user.education_level_code,
            "major_field_code":      user.major_field_code,
            "job_status_code":       user.job_status_code,
            "marital_status_code":   user.marital_status_code,
            "special_biz_code":      user.special_biz_code,
            "annual_income":         user.annual_income,
        }
        logger.info("유저 프로필 조회 완료 — user_id=%s, nickname=%s", user.user_id, user.nickname)

    response = await supervisor.run(user_inquiry=message, user_role=role, user_profile=user_profile)
    return JSONResponse(content={
        "conversation_id": conversation_id,
        "message": response,
    })