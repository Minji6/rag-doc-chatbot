import logging
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from ..state import ShareState

logger = logging.getLogger(__name__)

_vision_model = init_chat_model("gpt-4o", model_provider="openai", temperature=0.0)

_SYSTEM_PROMPT = (
    "당신은 청년 정책 챗봇의 이미지 분석기입니다. "
    "사용자가 첨부한 이미지에서 정책 상담에 관련된 정보를 추출하세요. "
    "공고문·신청서·계약서·증명서·화면 캡처 등 어떤 이미지든 분석합니다. "
    "추출한 정보는 한국어로 간결하게 정리하고, "
    "이미지에서 확인되는 정책명·지원 내용·신청 조건·날짜 등을 명시하세요."
)

# controller.py가 image_context를 사용자 메시지에 저장할 때 사용하는 구분자.
# 이 노드에서 복원 시 동일한 값으로 파싱한다.
IMAGE_CONTEXT_MARKER = "[이미지 분석 결과]\n"


def _restore_from_history(messages: list) -> str:
    """이전 대화 기록에서 이미지 분석 결과를 복원한다.

    controller.py는 이미지가 분석된 턴의 사용자 메시지를
    '{원본 질문}\n\n[이미지 분석 결과]\n{image_context}' 형태로 저장한다.
    이 함수는 해당 마커를 찾아 image_context 텍스트를 추출한다.
    """
    for m in reversed(messages):
        if isinstance(m, dict):
            role = m.get("role") or m.get("type") or ""
            content = str(m.get("content", ""))
        else:
            role = getattr(m, "type", "")
            content = str(getattr(m, "content", ""))
        if role in ("user", "human") and IMAGE_CONTEXT_MARKER in content:
            idx = content.find(IMAGE_CONTEXT_MARKER)
            extracted = content[idx + len(IMAGE_CONTEXT_MARKER):].strip()
            if extracted:
                logger.info("이전 대화에서 이미지 컨텍스트 복원 — %d자", len(extracted))
                return extracted
    return ""


async def image_analysis_node(state: ShareState) -> dict:
    """이미지를 GPT-4o Vision으로 분석해 텍스트 컨텍스트로 변환한다.

    - 이미지가 첨부된 경우: Vision API로 분석 후 image_context를 state에 저장.
    - 이미지가 없는 경우: 이전 대화 기록에서 image_context를 복원.
      이미지 관련 책임을 이 노드가 전담하므로, 하위 노드들은 state의
      image_context만 신뢰하면 된다.
    """
    image_base64 = state.get("image_base64")
    if not image_base64:
        restored = _restore_from_history(state.get("messages") or [])
        return {"image_context": restored}

    content_type = state.get("image_content_type") or "image/jpeg"
    logger.info("이미지 분석 노드 실행 — mime_type=%s", content_type)

    try:
        result = await _vision_model.ainvoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=[
                {"type": "text", "text": "첨부된 이미지에서 정책 상담에 도움이 될 정보를 추출해주세요."},
                {
                    "type": "image",
                    "base64": image_base64,
                    "mime_type": content_type,
                },
            ]),
        ])
        image_context = str(result.content).strip()
        logger.info("이미지 분석 완료 — %d자 추출", len(image_context))
        # 분석 완료 후 원본 제거 — checkpointer(PostgreSQL)에 base64 이미지가 영구 저장되는 것을 방지
        return {"image_context": image_context, "image_base64": None}
    except Exception as e:
        logger.warning("이미지 분석 실패 — 빈 컨텍스트로 계속 진행: %s", e)
        return {"image_context": "", "image_base64": None}
