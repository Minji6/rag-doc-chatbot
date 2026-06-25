# api/chat_service/langgraph/nodes/composer_node.py
"""Composer 노드 — 도메인 에이전트가 만든 본문 조각(fragment)을 하나의 답변으로 조립한다.

설계 (LangGraph supervisor-worker의 fan-in/aggregator 단계):
- 도메인 에이전트 4종은 "완성된 답변"이 아니라 "### 정책명 본문 조각"만 만든다.
- 이 노드가 cross-cutting 책임(인사·분야 헤더·맺음말)을 **단독으로** 진다.
  → 인사 N번 / 헤더 중복·누락 / 분야 간 형식 비일관 문제를 원천 차단.

LLM을 쓰지 않는다(결정적 조립). 인사는 고정 템플릿, 도입부는 분야 목록에서 파생.
→ 변동성 0, 토큰 0, gather 시절의 도입부 LLM 호출 1회도 제거.
"""
import logging
import re

from ..state import DomainResult, ShareState

logger = logging.getLogger(__name__)

_FALLBACK_MESSAGE = "죄송합니다, 답변을 생성하지 못했습니다. 다시 질문해주세요."

# 첫 대화에서만 1회 붙는 인사 템플릿. 후속 턴(대화기록 존재)에는 인사 없이 본문만.
# (대화가 이어지는 중에 매 턴 인사하면 어색하므로 conversation-start에만 부착)
_GREETING_TEMPLATE = "안녕하세요! 청년정책 안내를 도와드리는 챗봇입니다. 😊"

# 멀티 분야 답변을 항상 같은 순서로 정렬하기 위한 분야 출력 순서.
_CATEGORY_ORDER = ("주거", "일자리", "교육", "복지문화")

# 도메인 fragment에 인사/팀헤더/맺음말/분야헤더가 새어들어왔을 때 제거하는 방어 패턴.
# 1차 방어는 각 에이전트 프롬프트(OUTPUT_FORMAT_GUIDE)이고, 이 strip은 안전망이다.
_STRIP_PATTERNS = (
    re.compile(r"^[ \t]*#{1,3}[ \t]*(주거|일자리|교육|복지문화)[ \t]*$", re.MULTILINE),  # 분야 헤더
    re.compile(r"^.*정책팀[^\n]*[:：][ \t]*$", re.MULTILINE),                              # "🏠 주거정책팀 답변:"
    re.compile(r"^[ \t]*안녕하세요[^\n]*$", re.MULTILINE),                                  # 인사말
    re.compile(r"^[ \t]*감사합니다[.!]*[ \t]*$", re.MULTILINE),                             # 맺음말
    re.compile(r"^[ \t]*친절하게[^\n]*$", re.MULTILINE),                                    # "친절하게 안내..."
)


def _strip_boilerplate(text: str) -> str:
    """도메인 fragment에서 인사/팀헤더/맺음말/분야헤더를 제거한다 (안전망)."""
    for pattern in _STRIP_PATTERNS:
        text = pattern.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)  # 제거로 생긴 과도한 빈 줄 정리
    return text.strip()


def _collect_active_fragments(state: ShareState) -> list[DomainResult]:
    """텍스트가 있고 source != "none"인 도메인 결과만 분야 순서대로 정렬해 반환."""
    domain_results = state.get("domain_results") or {}
    active = [
        r for r in domain_results.values()
        if r.get("text") and r.get("source") != "none"
    ]
    order = {c: i for i, c in enumerate(_CATEGORY_ORDER)}
    active.sort(key=lambda r: order.get(r.get("category", ""), len(order)))
    return active


def _build_intro(categories: list[str]) -> str:
    """멀티 분야 도입부 1줄 (결정적 템플릿)."""
    return f"{', '.join(categories)} 분야의 정책을 안내해 드릴게요."


async def composer_node(state: ShareState) -> dict:
    """도메인 fragment를 모아 일관된 최종 답변으로 조립한다.

    분기:
    - 활성 결과 0개 → fallback (첫 턴이면 인사 포함)
    - 1개 → 분야 헤더 + 본문
    - N개 → 도입부 1줄 + 분야마다 헤더 + 본문
    인사 템플릿은 첫 대화(대화기록 없음)에서만 최상단에 1회 부착.
    """
    logger.info(
        "composer 노드 실행 — category=%s, inquiry_type=%s",
        state.get("category"), state.get("inquiry_type"),
    )

    suggestions = state.get("suggestions", [])
    fragments = _collect_active_fragments(state)

    # 대화기록(messages)이 비어 있으면 첫 턴 → 인사 템플릿 부착.
    is_first_turn = not state.get("messages")

    if not fragments:
        body = _FALLBACK_MESSAGE
        if is_first_turn:
            body = f"{_GREETING_TEMPLATE}\n\n{body}"
        return {"final_response": body, "suggestions": suggestions}

    parts: list[str] = []
    if is_first_turn:
        parts.append(_GREETING_TEMPLATE)
        parts.append("")

    if len(fragments) > 1:
        parts.append(_build_intro([f["category"] for f in fragments]))
        parts.append("")

    for frag in fragments:
        parts.append(f"## {frag['category']}")
        parts.append("")
        parts.append(_strip_boilerplate(frag["text"]))
        parts.append("")

    final_response = "\n".join(parts).rstrip() + "\n"
    return {"final_response": final_response, "suggestions": suggestions}
