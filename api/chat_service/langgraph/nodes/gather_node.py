# api/chat_service/langgraph/nodes/gather_node.py
import logging
from ..state import ShareState
from ..constants import CATEGORY_RESULT_FIELD

logger = logging.getLogger(__name__)

_FALLBACK_MESSAGE = "죄송합니다, 답변을 생성하지 못했습니다. 다시 질문해주세요."


async def gather_node(state: ShareState) -> dict:
    # NOTE: task #4(다중 결과 병합/비교)에서 본격 재작성 예정. 현재는 단일 분야 패스스루.
    logger.info("결과 취합 노드 실행 — category=%s", state.get("category"))

    field_name = CATEGORY_RESULT_FIELD.get(state.get("category", ""))
    domain_result = state.get(field_name) if field_name else None
    text = domain_result.get("text", "") if domain_result else ""

    final_response = text if text else _FALLBACK_MESSAGE
    return {"final_response": final_response}
