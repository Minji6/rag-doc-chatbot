# api/chat_service/langgraph/nodes/gather_node.py
import logging

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from ..state import DomainResult, ShareState

logger = logging.getLogger(__name__)

_FALLBACK_MESSAGE = "죄송합니다, 답변을 생성하지 못했습니다. 다시 질문해주세요."

# 멀티 분야 도입부 생성용 — 모듈 싱글톤. 짧은 1~2줄 작성이라 토큰 비용 미미.
_intro_model = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0.3)

_INTRO_SYSTEM_PROMPT = (
    "당신은 청년정책 챗봇의 답변 도입부 작성기입니다. "
    "여러 분야에 걸친 질문에 대해 **반드시 한 문장**짜리 짧은 도입부를 작성하세요.\n\n"
    "규칙:\n"
    "- **분야가 N개여도 모두 한 문장에 담을 것.** 분야마다 문장을 나누지 마세요.\n"
    "- 분야명을 그대로 자연스럽게 나열 (예: '주거와 일자리', '주거·일자리·교육')\n"
    "- 헤더(#, ##), 정책 내용, 이모지, 인사말('안녕하세요') 금지.\n"
    "- 줄바꿈 금지. 한 줄, 한 문장.\n\n"
    "예시 (2분야): 주거와 일자리 분야 정책을 함께 안내드립니다.\n"
    "예시 (3분야): 주거, 일자리, 교육 분야 정책을 안내드립니다.\n"
    "예시 (4분야): 주거, 일자리, 교육, 복지문화 분야 정책을 안내드립니다."
)


def _collect_active_results(state: ShareState) -> list[DomainResult]:
    """state.domain_results에서 텍스트가 있고 source != "none"인 결과만 수집."""
    domain_results = state.get("domain_results") or {}
    return [
        r for r in domain_results.values()
        if r.get("text") and r.get("source") != "none"
    ]


async def _build_multi_domain_intro(categories: list[str]) -> str:
    """멀티 분야 답변 도입부를 LLM으로 생성. 실패 시 고정 문구 fallback."""
    try:
        result = await _intro_model.ainvoke([
            SystemMessage(content=_INTRO_SYSTEM_PROMPT),
            HumanMessage(content=f"분야 목록: {', '.join(categories)}"),
        ])
        return str(result.content).strip()
    except Exception as e:
        logger.warning("도입부 생성 실패 — 고정 문구 사용: %s", e)
        return (
            f"질문이 {', '.join(categories)} 분야에 걸쳐 있어 "
            f"각 분야 답변을 함께 안내해드립니다."
        )


async def gather_node(state: ShareState) -> dict:
    """도메인 결과를 모아 최종 답변을 작성한다.

    분기:
    - 0개 결과 → fallback 메시지
    - 1개 결과 → 단일 분야 텍스트 (## 헤더는 단일 분야엔 생략, 도메인 자체 인사말이 있음)
    - N개 결과 → LLM 도입부 1~2줄 + 분야별 "## {category}" 헤더 + text concat
    - inquiry_type == "비교" → 추후 구현 (현재는 일반 흐름과 동일)
    """
    inquiry_type = state.get("inquiry_type", "")
    logger.info("결과 취합 노드 실행 — category=%s, inquiry_type=%s",
                state.get("category"), inquiry_type)

    # TODO: inquiry_type == "비교" 케이스 별도 처리
    # - 활성 도메인의 policies(raw 메타)를 모아 마크다운 표로 가공
    # - LLM이 |항목|정책A|정책B|... 형식 표 작성
    # - 현재는 일반 흐름으로 패스스루

    results = _collect_active_results(state)
    suggestions = state.get("suggestions", [])

    if not results:
        return {"final_response": _FALLBACK_MESSAGE, "suggestions": suggestions}

    if len(results) == 1:
        # 단일 분야 — 도메인 텍스트 그대로 (각 에이전트가 자체 인사말 포함)
        return {"final_response": results[0]["text"], "suggestions": suggestions}

    # 멀티 분야 — LLM 도입부 + 분야별 ## 헤더 + concat
    categories = [r["category"] for r in results]
    intro = await _build_multi_domain_intro(categories)

    parts = [intro, ""]
    for r in results:
        parts.append(f"## {r['category']}")
        parts.append(r["text"])
        parts.append("")  # 분야 간 공백 줄

    final_response = "\n".join(parts).rstrip() + "\n"
    return {"final_response": final_response, "suggestions": suggestions}
