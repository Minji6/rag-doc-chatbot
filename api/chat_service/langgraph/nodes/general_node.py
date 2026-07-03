# api/chat_service/langgraph/nodes/general_node.py
"""일반대화 노드 — 정책과 무관한 인사·잡담·일반 질문에 응대한다.

설계:
- analysis_node가 is_general=true로 판정한 질문만 route_node_fun이 이 노드로 보낸다.
- 정책 검색(RAG)을 전혀 돌리지 않고, 단일 LLM 호출로 자연스러운 대화를 생성한다.
- 정책봇 정체성은 유지한다: 가볍게 응대하되, 도움 줄 수 있는 정책 주제로 부드럽게 유도.
- composer를 거치지 않고 final_response를 직접 채워 바로 END로 간다. policies는 없다.
- 게스트(비로그인)가 '추천' 의도로 물을 때도 route_node_fun이 이 노드로 보낸다.
  이 경우엔 LLM 호출 없이 회원가입 유도 고정 멘트로 바로 응답한다(검색·생성 비용 없음).
"""
import logging

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

from ..constants import ROLE_GUEST
from ..state import ShareState

logger = logging.getLogger(__name__)

_GUEST_RECOMMEND_MESSAGE = (
    "정책 추천은 회원가입 후 이용하실 수 있는 기능이에요! 🙌\n"
    "로그인하시면 회원님의 조건에 꼭 맞는 정책을 찾아 추천해드릴게요.\n"
    "지금은 궁금하신 분야나 정책명을 알려주시면 검색은 바로 도와드릴 수 있어요."
)

# 모듈 싱글톤 — 한 번만 생성
_chat_model = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0.5)

_SYSTEM_PROMPT = """당신은 청년정책을 안내하는 친근한 챗봇입니다.
지금 사용자의 메시지는 정책 검색이 필요 없는 인사·잡담·가벼운 일반 질문입니다.

규칙:
- 따뜻하고 자연스럽게, 1~3문장으로 짧게 응대하세요.
- 당신이 도울 수 있는 분야는 청년 **일자리·주거·교육·복지문화** 정책입니다.
  대화 흐름이 자연스러우면 이 주제로 가볍게 안내·유도하세요. (매번 억지로 끼워넣지는 말 것)
- 정책 정보(금액·자격·마감 등)를 지어내지 마세요. 구체 정책은 "찾아드릴까요?"처럼 제안만 하세요.
- [이미지 분석 결과]가 제공된 경우, 해당 내용은 시스템이 이미 추출한 사실입니다.
  사용자가 그 내용(학점·소득 등)을 묻는다면 [이미지 분석 결과]를 근거로 답변하세요.
- 날씨·뉴스·실시간 시세·현재 시각 등 당신이 알 수 없는 실시간 정보는 아는 척 지어내지 마세요.
  모른다고 솔직히 말한 뒤, 도와드릴 수 있는 청년정책 주제로 부드럽게 돌리세요.
- 첫 대화라면 가벼운 자기소개를 곁들여도 좋습니다."""

# 최근 대화 맥락으로 넘길 메시지 수 (잡담 응대엔 짧게 충분)
_RECENT_TURNS = 6


def _history_text(messages: list) -> str:
    """최근 대화를 LLM 입력용 텍스트로 압축한다 (dict / LangChain 메시지 객체 모두 지원)."""
    lines: list[str] = []
    for m in messages[-_RECENT_TURNS:]:
        if isinstance(m, dict):
            role = m.get("role") or m.get("type") or ""
            content = str(m.get("content", ""))
        else:
            role = getattr(m, "type", "")
            content = str(getattr(m, "content", ""))
        speaker = "사용자" if role in ("user", "human") else "챗봇"
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


async def general_node(state: ShareState) -> dict:
    """정책 무관 일반대화 또는 게스트 추천 차단에 응대해 final_response를 직접 채운다 (composer 우회)."""
    # 게스트 추천 차단 — is_general=false로 들어오지만(정책 관련 질문은 맞음) '추천'이라 검색 없이 회원가입 안내.
    if state.get("user_role") == ROLE_GUEST and "추천" in (state.get("inquiry_type") or []):
        logger.info("게스트 추천 차단 — query=%s", state["user_inquiry"][:30])
        return {"final_response": _GUEST_RECOMMEND_MESSAGE, "suggestions": []}

    inquiry = state["user_inquiry"]
    messages = state.get("messages") or []
    image_context = state.get("image_context") or ""
    logger.info("일반대화 노드 실행 — query=%s", inquiry[:30])

    parts: list[str] = []
    history = _history_text(messages)
    if history:
        parts.append(f"[최근 대화]\n{history}")
    if image_context:
        parts.append(f"[이미지 분석 결과]\n{image_context}")
    parts.append(f"[사용자 새 메시지]\n{inquiry}")
    user_content = "\n\n".join(parts)

    try:
        result = await _chat_model.ainvoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ])
        text = str(result.content).strip()
    except Exception:
        logger.exception("일반대화 생성 실패 — 폴백 메시지 사용")
        text = (
            "안녕하세요! 청년정책 안내를 도와드리는 챗봇입니다. 😊\n"
            "일자리·주거·교육·복지문화 관련해 궁금한 점이 있으면 편하게 물어봐 주세요."
        )

    return {"final_response": text, "suggestions": []}
