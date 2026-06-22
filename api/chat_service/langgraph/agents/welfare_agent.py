import json
import logging
from typing import Annotated, Literal
from fastapi import Depends
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from ..state import DomainResult
from ..constants import AGENT_CATEGORY
from tools.common.tool_rag_search import search_web_supplement
from tools.common.tool_compare import compare_policies, policy_priority_score
from tools.common.tool_dday import calculate_dday
from tools.common.tool_priority import answer_with_priority
from tools.welfare.tool_user_profile import build_profile_context, check_missing_profile
from tools.welfare.tool_eligibility import check_eligibility

logger = logging.getLogger(__name__)

_CATEGORY = AGENT_CATEGORY["welfare"]

_EXTRACT_SYSTEM_PROMPT = """대화에서 사용자가 직접 알려준 프로필 정보만 JSON으로 추출하세요.
추출할 필드 (언급이 없거나 모른다고 하면 반드시 null):
- jobcd: 취업상태 ("미취업" / "재직" / "자영업", 언급 없으면 null)
- earncndsecd: 소득분위 (1~10 정수. "모르겠다"·"연 소득 X원" 등 분위 불명확하면 null)
- zipcd: 거주지역 (시/도 이름, 언급 없으면 null)
- mrgsttscd: 혼인여부 ("미혼" / "기혼" / "이혼", 언급 없으면 null)
- schoolcd: 학력 ("고졸" / "대학재학" / "대학졸업" 등, 언급 없으면 null)
- sbizcd: 특수분류 ("장애" / "한부모" / "다문화" / "해당없음", 언급 없으면 null)
JSON만 반환하세요. 다른 텍스트 없이."""


async def _extract_profile_from_history(history: list) -> dict:
    """대화 히스토리에서 사용자가 직접 알려준 프로필 값을 LLM으로 추출합니다."""
    if not history:
        return {}

    def _content(m) -> str:
        if isinstance(m, dict):
            return m.get("content", "")
        return getattr(m, "content", "")

    def _role(m) -> str:
        if isinstance(m, dict):
            return "사용자" if m.get("role") == "user" else "AI"
        t = getattr(m, "type", "")
        return "사용자" if t == "human" else "AI"

    recent = history[-10:]
    conversation = "\n".join(f"{_role(m)}: {_content(m)}" for m in recent)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = await llm.ainvoke([
        SystemMessage(content=_EXTRACT_SYSTEM_PROMPT),
        HumanMessage(content=conversation),
    ])

    try:
        text = str(response.content).strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text)
        extracted = {k: v for k, v in data.items() if v is not None}
        if extracted:
            logger.info("대화에서 추출한 프로필: %s", extracted)
        return extracted
    except Exception as e:
        logger.warning("프로필 추출 실패: %s", e)
        return {}


def _resolve_source(messages: list) -> Literal["rag", "web", "none"]:
    """agent 실행 후 tool 호출 이력을 보고 source를 결정합니다."""
    called_tools = {
        msg.name
        for msg in messages
        if hasattr(msg, "name") and msg.name
    }
    if "search_web_supplement" in called_tools:
        return "web"
    if "answer_with_priority" in called_tools:
        return "rag"
    return "none"

_BASE_SYSTEM_PROMPT = """당신은 청년 복지문화 정책 전문가입니다. 복지문화 관련 정책만 안내하세요.

## 도구 사용 규칙

### answer_with_priority (항상 첫 번째)
- 질문이 들어오면 반드시 가장 먼저 호출하세요.
- category는 항상 "복지문화"로 설정하세요.
- "[RAG]\n..." 형식으로 RAG 결과를 반환하며, RAG가 부족하면 "[LLM]\n..." 형식으로 웹 결과가 추가됩니다.
- "[RAG]" 섹션의 정책을 "[LLM]" 섹션보다 우선하여 사용자에게 안내하세요.

### search_web_supplement (사용자가 추가 검색 요청 시)
- 사용자가 "더 없어?", "다른 것도 찾아줘" 등 추가 검색을 명시적으로 요청할 때만 호출하세요.
- exclude_titles에 이미 안내한 정책명을 넣어 중복을 방지하세요.

### calculate_dday (마감일 관련 질문 시)
- 사용자가 마감일을 물을 때 호출하세요.
- deadline: bizPrdEndYmd 값, apply_period_type: aplyPrdSeCd 값을 사용하세요.
- 마감일 데이터가 없으면 호출하지 마세요.

### compare_policies (비교 요청 시)
- 사용자가 "비교해줘"라고 하면 반드시 호출하세요.
- policy_list에 "[RAG]" 섹션에서 파악한 정책 정보를 딕셔너리 목록으로 구성해서 전달하세요.
- "[RAG]" 결과가 없는 경우에는 호출하지 말고 텍스트로 비교해서 안내하세요.

### policy_priority_score (추천 요청 시)
- 사용자가 "추천해줘"라고 하면 최종 답변 전에 호출하세요.
- policy_list에 "[RAG]" 섹션에서 파악한 정책 정보를 딕셔너리 목록으로 구성해서 전달하세요.

### check_eligibility (자격 확인 요청 시)
- 사용자가 "신청 가능해?", "자격 되나?" 등을 물을 때만 호출하세요.
- policy_metadata에 해당 정책의 정보를 딕셔너리로 구성해서 전달하세요.

## 사용자 프로필 안내 지침
- 아래 [사용자 프로필]에 정보가 있으면 프로필에 맞는 정책을 우선순위 높여 설명하세요.
- guest(비로그인)인 경우 일반적인 복지 정책을 안내하세요."""


class WelfareAgent:
    def __init__(self, model: str = "openai:gpt-4o-mini") -> None:
        self.logger = logging.getLogger(f"{__name__}.WelfareAgent")
        self._model = model

    def _build_system_prompt(self, user_role: str, user_profile: dict, inquiry_type: str) -> str:
        parts = [_BASE_SYSTEM_PROMPT]
        if inquiry_type:
            parts.append(f"## 현재 질문 의도\n사용자의 질문 의도는 '{inquiry_type}'입니다. 이에 맞는 도구를 우선 사용하세요.")
        profile_context = build_profile_context(user_role, user_profile)
        if profile_context:
            parts.append(profile_context)
        return "\n\n".join(parts)

    def _make_agent(self, user_role: str, user_profile: dict, inquiry_type: str):
        system_prompt = self._build_system_prompt(user_role, user_profile, inquiry_type)
        return create_agent(
            model=self._model,
            tools=[
                answer_with_priority,
                search_web_supplement,
                calculate_dday,
                policy_priority_score,
                compare_policies,
                check_eligibility,
            ],
            system_prompt=system_prompt,
        )

    async def run(
        self,
        question: str,
        user_role: str = "guest",
        user_profile: dict | None = None,
        inquiry_type: str = "",
        history: list | None = None,
    ) -> DomainResult:
        user_profile = user_profile or {}

        # 대화 히스토리에서 사용자가 알려준 프로필 정보 추출 후 DB 프로필에 병합
        if user_role == "user" and history:
            extracted = await _extract_profile_from_history(history)
            if extracted:
                user_profile = {**user_profile, **extracted}

        agent = self._make_agent(user_role, user_profile, inquiry_type)

        # 이전 대화 히스토리 + 현재 질문을 함께 전달
        messages = list(history or [])
        messages.append({"role": "user", "content": question})

        result = await agent.ainvoke({"messages": messages})

        text = result["messages"][-1].content
        source = _resolve_source(result["messages"])

        # 이미 이전 대화에서 유도 질문을 보낸 적 있으면 반복하지 않음
        already_asked = any(
            "더 정확한 복지 정책을 안내해 드리기 위해" in str(getattr(m, "content", m))
            for m in (history or [])
        )
        if user_role == "user" and user_profile and not already_asked:
            missing_fields, follow_up = check_missing_profile(user_profile)
            if missing_fields:
                text = f"{text}\n\n{follow_up}"

        return DomainResult(text=text, policies=[], category=_CATEGORY, source=source)


WelfareAgentDep = Annotated[WelfareAgent, Depends(WelfareAgent)]
