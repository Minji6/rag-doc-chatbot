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
from ._policy_table import build_comparison_table

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


def _build_domain_comments(fragments: list[DomainResult]) -> str:
    """비교 모드 — 각 도메인 에이전트가 만든 정성 코멘트를 분야별로 모은다 (하이브리드).

    표(정형 데이터)는 composer가, 코멘트(분야 판단)는 에이전트가 책임진다.
    혹시 에이전트가 표를 흘렸어도 "|"로 시작하는 표 줄은 안전망으로 제거한다.
    """
    lines: list[str] = []
    for frag in fragments:
        comment = _strip_boilerplate(frag.get("text", ""))
        # 표 줄(| ... |)이 새어들어왔으면 제거 — 표는 위에서 이미 결정적으로 그렸음.
        comment = "\n".join(
            ln for ln in comment.splitlines() if not ln.lstrip().startswith("|")
        ).strip()
        if comment:
            lines.append(f"- **{frag.get('category', '')}**: {comment}")
    if not lines:
        return ""
    return "**분야별 코멘트**\n" + "\n".join(lines)


def _collect_policies_for_compare(fragments: list[DomainResult]) -> list[dict]:
    """활성 도메인 결과에서 raw 정책 메타를 분야 순서대로 평탄화한다.

    각 정책에 분야 라벨을 보강한다(메타에 category가 없을 때 도메인 결과 category로 채움).
    """
    policies: list[dict] = []
    for frag in fragments:
        domain_category = frag.get("category", "")
        for policy in frag.get("policies", []):
            if not policy:
                continue
            if not policy.get("category"):
                policy = {**policy, "category": domain_category}
            policies.append(policy)
    return policies


async def composer_node(state: ShareState) -> dict:
    """도메인 fragment를 모아 일관된 최종 답변으로 조립한다.

    분기:
    - 활성 결과 0개 → fallback (첫 턴이면 인사 포함)
    - inquiry_type=="비교" & 정책 2개 이상 → 결정적 통합 비교 표 (분야 교차 가능)
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

    # 비교 의도 → 분야 fragment를 쌓는 대신 raw 정책 메타로 통합 비교 표를 결정적으로 조립.
    # (주거+일자리처럼 분야가 섞여도 하나의 표로 나란히 비교. 정책 2개 미만이면 일반 분기로 폴백.)
    comparison_table = ""
    if state.get("inquiry_type") == "비교":
        # 후속질문이 직전 정책을 지정했으면(resolved_policies) 검색 결과 대신 '그 정책들'로 표를 만든다.
        # → "공공근로와 미래일자리 비교"가 유사도 top-k가 아니라 정확히 지정 2개만 비교됨.
        resolved = state.get("resolved_policies") or []
        compare_policies = resolved if len(resolved) >= 2 else _collect_policies_for_compare(fragments)
        comparison_table = build_comparison_table(compare_policies)

    if comparison_table:
        categories = [f["category"] for f in fragments]
        parts.append(f"{', '.join(categories)} 분야 정책을 비교해 드릴게요." if len(categories) > 1
                     else f"{categories[0]} 정책을 비교해 드릴게요.")
        parts.append("")
        parts.append(comparison_table)
        # 하이브리드: 결정적 표 아래에 각 도메인 에이전트의 정성 코멘트 배치.
        domain_comments = _build_domain_comments(fragments)
        if domain_comments:
            parts.append("")
            parts.append(domain_comments)
    else:
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
