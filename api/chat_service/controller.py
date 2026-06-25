import base64
import logging
from typing import Annotated
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from api.chat_service.langgraph.supervisor import ChatbotSupervisorDep
from api.chat_service.langgraph.constants import ROLE_USER, ROLE_GUEST
from api.auth_service.service import UserServiceDep
from api.common.sqlalchemy_conf import OrmSessionDep
from api.history_service.agent_dependency import HistoryAgentByFormDep, build_thread_id

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])

_MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10MB


@router.post("/service", response_class=JSONResponse)
async def chat(
    message: Annotated[str, Form()],
    conversation_id: Annotated[str, Form()],
    role: Annotated[str, Form()],
    supervisor: ChatbotSupervisorDep,
    user_service: UserServiceDep,
    session: OrmSessionDep,
    agent: HistoryAgentByFormDep, # role에 따라 자동 주입
    user_id: Annotated[int | None, Form()] = None,
    attach: Annotated[UploadFile | None, File()] = None,
):
    if role not in (ROLE_USER, ROLE_GUEST):
        raise HTTPException(status_code=422, detail=f"role은 '{ROLE_USER}' 또는 '{ROLE_GUEST}'여야 합니다.")

    logger.info(f"[{conversation_id}] 메시지 수신: {message[:30]}...")

    user_profile = None # guest면 None
    if role == ROLE_USER:
        if not user_id:
            raise HTTPException(status_code=422, detail="user role에는 user_id가 필요합니다.")
        user_profile = await user_service.get_user_profile(user_id, session)
        if not user_profile:
            raise HTTPException(status_code=404, detail=f"user_id={user_id} 유저를 찾을 수 없습니다.")

    image_base64 = None
    image_content_type = None
    if attach:
        if not (attach.content_type or "").startswith("image/"):
            raise HTTPException(status_code=415, detail="이미지 파일만 첨부 가능합니다.")
        image_data = await attach.read()
        if len(image_data) > _MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="이미지 파일은 10MB 이하만 허용됩니다.")
        image_base64 = base64.b64encode(image_data).decode("utf-8")
        image_content_type = attach.content_type
        logger.info(f"[{conversation_id}] 이미지 첨부 수신: {attach.filename} ({len(image_data)} bytes)")

    # user → "{user_id}:{conversation_id}", guest → "{conversation_id}"
    thread_id = build_thread_id(role, conversation_id, str(user_id) if user_id else None)

    # 이전 대화 불러오기 - 첫 대화면 빈 리스트 반환
    history = await agent.get_history(thread_id)
    logger.info(f"[{thread_id}] 이전 대화 {len(history['messages'])}개 로드")

    result = await supervisor.run(
        user_inquiry=message,
        user_role=role,
        user_profile=user_profile,
        messages=history["messages"],
        image_base64=image_base64,
        image_content_type=image_content_type,
    )

    # 새 대화 저장 - LLM에 재호출 없이 checkpointer에 직접 저장 (저장은 message 본문만)
    await agent.save_exchange(message, result["message"], thread_id)
    logger.info(f"[{thread_id}] 대화 저장 완료")

    return JSONResponse(content={
        "conversation_id": conversation_id,
        **result,   # message, category, inquiry_type, policies, suggestions
    })