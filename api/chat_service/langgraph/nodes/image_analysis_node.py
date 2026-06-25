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


async def image_analysis_node(state: ShareState) -> dict:
    """이미지를 GPT-4o Vision으로 분석해 텍스트 컨텍스트로 변환한다.

    이미지가 없으면 즉시 빈 문자열을 반환하므로 이미지 없는 요청에서도 비용이 발생하지 않는다.
    """
    image_base64 = state.get("image_base64")
    if not image_base64:
        return {"image_context": ""}

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
