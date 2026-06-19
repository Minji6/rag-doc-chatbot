# api/chat_service/langgraph/nodes/gather_node.py
import logging
from ..state import ShareState
from ..constants import CATEGORY_RESULT_FIELD

logger = logging.getLogger(__name__)

_FALLBACK_MESSAGE = "죄송합니다, 답변을 생성하지 못했습니다. 다시 질문해주세요."


async def gather_node(state: ShareState) -> dict:
    logger.info("결과 취합 노드 실행 — category=%s", state.get("category"))

    field_name = CATEGORY_RESULT_FIELD.get(state.get("category", ""))
    result = state.get(field_name, "") if field_name else ""

    final_response = result if result else _FALLBACK_MESSAGE
    return {"final_response": final_response}   # 변경된 키만 반환 (지침서 18-2)