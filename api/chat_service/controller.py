import base64
import logging
from typing import Annotated
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from api.chat_service.langgraph.supervisor import ChatbotSupervisorDep
from api.chat_service.langgraph.constants import ROLE_USER
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
    # role 값 검증(user/guest)은 의존성(HistoryAgentByFormDep, role: Literal)에서 먼저 수행되므로
    # 여기서는 별도 체크 없이 진입한다.
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

    # 직전 정책 메모리(후속질문 해소용)의 생명주기는 supervisor가 thread_id로 직접 관리한다.
    result = await supervisor.run(
        user_inquiry=message,
        user_role=role,
        user_profile=user_profile,
        messages=history["messages"],
        image_base64=image_base64,
        image_content_type=image_content_type,
        thread_id=thread_id,
    )

    # 새 대화 저장 - LLM에 재호출 없이 checkpointer에 직접 저장.
    # full_message는 의도 정형 전 전체 답변 — 상세조회처럼 응답 message를 비워도
    # 대화 맥락(후속질문 해소)은 보존되도록 히스토리엔 전체 답변을 저장한다.
    history_message = result.pop("full_message", "") or result["message"]
    # image_context가 있으면 사용자 메시지에 포함해 저장한다.
    # 후속 턴의 contextualize_node / general_node가 messages를 통해 이미지 분석 결과를 참조할 수 있도록.
    image_context = result.pop("image_context", "") or ""
    history_user_message = f"{message}\n\n[이미지 분석 결과]\n{image_context}" if image_context else message
    await agent.save_exchange(history_user_message, history_message, thread_id)
    logger.info(f"[{thread_id}] 대화 저장 완료")

    return JSONResponse(content={
        "conversation_id": conversation_id,
        **result,   # message, category, inquiry_type, policies, suggestions
    })