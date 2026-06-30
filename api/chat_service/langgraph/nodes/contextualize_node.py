"""대화 맥락화 노드 — 후속질문을 독립형 질의로 재작성하고, 가리킨 직전 정책을 해소한다.

이 노드가 없으면 파이프라인은 매 턴 현재 문장만으로 stateless하게 동작한다.
("두번째 정책 알려줘"를 글자 그대로 임베딩 검색 → 엉뚱한 결과). 이 노드는 supervisor로
주입된 messages(대화기록)와 last_policies(직전 턴 정책)를 활용해 두 가지를 한다:

L1) 질의 재작성: "두번째 정책 알려줘" → "공공근로·미래일자리 등 중 두번째인 ○○ 정책 상세 안내"
    → 재작성된 standalone 질의를 user_inquiry에 덮어써 하위 모든 노드(analysis/search/생성)가 활용.
L2) 참조 해소: 사용자가 가리킨 직전 정책(순번/이름)을 last_policies에서 골라 resolved_policies로.
    → composer가 비교 시 검색 결과 대신 '지정한 그 정책'으로 표를 그리게 한다(정확성).

첫 턴(대화기록·직전정책 모두 없음)에는 LLM을 호출하지 않고 그대로 통과한다(토큰 0).
"""
import logging
from typing import Annotated, cast

from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

from ..state import ShareState

logger = logging.getLogger(__name__)

# 재작성 LLM에 넘길 직전 대화 메시지 수 (너무 길면 비용↑·노이즈↑)
_RECENT_TURNS = 6
# AI 답변은 비교 표 등으로 매우 길 수 있어 잘라서 전달 (참조 해소엔 앞부분이면 충분)
_AI_MSG_CLIP = 400


class Contextualized(BaseModel):
    """후속질문 맥락화 결과 구조화 출력."""
    standalone_query: Annotated[
        str,
        Field(description=(
            "대화 맥락이 모두 풀린 독립형 한국어 질의. 지시대명사/순번('이거','두번째')을 "
            "구체 정책명으로 치환. 후속 맥락이 필요 없는 새 질문이면 원문을 거의 그대로 둔다."
        ))
    ]
    referenced_indices: Annotated[
        list[int],
        Field(description=(
            "사용자가 명시적으로 가리킨 [직전 정책] 목록의 1-based 번호들. "
            "예: '두번째 정책'→[2], '공공근로와 미래일자리 비교'→해당 두 정책 번호. "
            "직전 정책을 가리키지 않는 새 검색이면 빈 리스트 []."
        ))
    ]


_chat_model = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0.0)
_structured_model = _chat_model.with_structured_output(Contextualized)

_SYSTEM_PROMPT = """당신은 청년정책 챗봇의 대화 맥락 분석기입니다.
사용자의 새 메시지가 **이전 대화에 의존하는 후속질문**인지 보고, 두 가지를 수행하세요.

1) standalone_query: 이전 대화 맥락을 모두 반영한 독립형 질의로 다시 쓰세요.
   - "두번째 정책 알려줘", "그거 자세히", "방금 거랑 비교" 같은 표현의 지시대상을
     [직전 정책] 목록에서 찾아 **구체 정책명**으로 바꿔 쓰세요.
   - 맥락이 필요 없는 완전히 새로운 질문이면 원문을 거의 그대로 두세요.
   - [첨부 이미지 분석 결과]가 있을 때, 사용자가 그 안의 수치(학점·소득 등)를 묻는 질문
     ("내 학점이 몇이야?", "내 소득이 얼마야?" 등)은 **원문을 그대로** standalone_query로 쓰세요.
     "어떻게 확인하나요?" 같은 방법 질문으로 바꾸지 마세요.
2) referenced_indices: 사용자가 가리킨 [직전 정책]의 1-based 번호. 없으면 [].
   - "방금/위에서 알려준 것들", "아까 그거 전부", "다 비교해줘"처럼 직전 목록 **전체**를
     가리키면 [직전 정책]의 **모든 번호**를 담으세요. (일부만 임의로 고르지 말 것)

추측하지 말고 [직전 정책]에 실제로 있는 항목만 번호로 고르세요."""


def _msg_role_content(m) -> tuple[str, str]:
    """메시지에서 (role, content)를 추출. dict와 LangChain 메시지 객체 모두 지원.

    state["messages"]는 add_messages 리듀서가 dict를 HumanMessage/AIMessage 객체로
    변환해 보관하므로(.get 없음), 두 형태를 모두 받아야 한다.
    """
    if isinstance(m, dict):
        return (m.get("role") or m.get("type") or ""), str(m.get("content", ""))
    return getattr(m, "type", ""), str(getattr(m, "content", ""))


def _recent_messages_text(messages: list) -> str:
    """최근 대화 몇 턴을 LLM 입력용 텍스트로 압축한다."""
    recent = messages[-_RECENT_TURNS:]
    lines = []
    for m in recent:
        role, content = _msg_role_content(m)
        speaker = "사용자" if role in ("user", "human") else "챗봇"
        if speaker == "챗봇" and len(content) > _AI_MSG_CLIP:
            content = content[:_AI_MSG_CLIP] + " …(생략)"
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


def _format_last_policies(last_policies: list[dict]) -> str:
    """직전 정책을 번호 매긴 목록으로 (referenced_indices 매핑 기준)."""
    lines = []
    for i, p in enumerate(last_policies, 1):
        name = p.get("plcyNm", "(이름 없음)")
        cat = p.get("category", "")
        lines.append(f"{i}. {name}" + (f" [{cat}]" if cat else ""))
    return "\n".join(lines)


async def contextualize_node(state: ShareState) -> dict:
    """후속질문을 독립형 질의로 재작성하고 참조 정책을 해소한다.

    Returns(부분 업데이트):
        - user_inquiry: 재작성된 standalone 질의 (덮어쓰기)
        - resolved_policies: 사용자가 가리킨 직전 정책 부분집합 (없으면 [])
    """
    messages = state.get("messages") or []
    last_policies = state.get("last_policies") or []
    original = state["user_inquiry"]

    # 첫 턴 — 맥락이 없으면 LLM 없이 통과 (토큰 절약).
    # resolved_policies는 매 턴 여기서 새로 세팅된다(초기 state는 supervisor가 []로 시드).
    # → 주제를 바꾼 새 질문엔 [](아래 LLM이 referenced_indices=[] 반환)로 덮여 스테일 라우팅이 없다.
    if not messages and not last_policies:
        return {"resolved_policies": []}

    image_context = (state.get("image_context") or "").strip()
    image_hint = f"[첨부 이미지 분석 결과]\n{image_context[:300]}\n\n" if image_context else ""
    user_content = (
        f"[직전 정책]\n{_format_last_policies(last_policies) or '(없음)'}\n\n"
        f"[최근 대화]\n{_recent_messages_text(messages) or '(없음)'}\n\n"
        f"{image_hint}"
        f"[사용자 새 메시지]\n{original}"
    )

    try:
        result = cast(Contextualized, await _structured_model.ainvoke([
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]))
    except Exception:
        # 실패 시 user_inquiry를 반환에 넣지 않음 → state의 원문 질의가 그대로 유지(LangGraph 부분 업데이트).
        # resolved_policies만 []로 비워 직전 참조가 라우팅에 새지 않게 한다.
        logger.exception("맥락화 실패 — 원문 질의로 폴백")
        return {"resolved_policies": []}

    # 1-based 번호 → 직전 정책 dict 매핑 (범위 밖 번호는 무시)
    resolved = [
        last_policies[i - 1]
        for i in result.referenced_indices
        if 1 <= i <= len(last_policies)
    ]

    rewritten = result.standalone_query.strip() or original
    logger.info(
        "맥락화 — 원문=%r → 재작성=%r, 참조정책=%d건",
        original[:30], rewritten[:50], len(resolved),
    )
    return {"user_inquiry": rewritten, "resolved_policies": resolved}
