import json
import logging
from typing import Annotated, Literal
from fastapi import Depends
from langchain.agents import create_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.chat_models import init_chat_model
from ..constants import AGENT_CATEGORY, OUTPUT_FORMAT_GUIDE, COMPARISON_COMMENT_GUIDE, OUTPUT_DETAIL_GUIDE, RECOMMEND_GUIDE
from ..tools.dday import calculate_dday
from ..tools.age import calc_age as _calc_age
from ..tools.eligibility import is_eligibility_inquiry  # type: ignore[attr-defined]
from ..tools import SUGGESTIONS_PROMPT, parse_suggestions

logger = logging.getLogger(__name__)

_CATEGORY = AGENT_CATEGORY["welfare"]
_MIN_RAG_COUNT = 5  # RAG 결과가 이 수 미만이면 웹 검색으로 보완

_profile_extractor = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0)


##############################################################
# 유틸
##############################################################

def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _build_profile_context(user_profile: dict) -> str:
    if not user_profile:
        return ""

    age = _calc_age(str(user_profile.get("birth_date", "")))

    def v(val):
        return val if val not in (None, "", "제한없음") else "미입력"

    lines = ["[사용자 프로필]"]
    lines.append(f"  나이        : {age}세" if age else "  나이        : 미입력")
    lines.append(f"  거주 지역   : {v(user_profile.get('zipcd'))}")
    lines.append(f"  취업 상태   : {v(user_profile.get('jobcd'))}")
    lines.append(f"  소득 분위   : {v(user_profile.get('earncndsecd'))}")
    lines.append(f"  혼인 여부   : {v(user_profile.get('mrgsttscd'))}")
    lines.append(f"  학력        : {v(user_profile.get('schoolcd'))}")
    lines.append(f"  특수 분류   : {v(user_profile.get('sbizcd'))}")
    return "\n".join(lines)


def _msg_type(msg) -> str:
    return getattr(msg, "type", None) or msg.get("type", msg.get("role", ""))

def _msg_content(msg) -> str:
    c = getattr(msg, "content", None) if hasattr(msg, "content") else msg.get("content", "")
    return c if isinstance(c, str) else ""

def _build_agent_history(messages: list) -> list[HumanMessage | AIMessage]:
    """LangGraph 에이전트에 전달할 히스토리. human/ai 메시지만 최근 6개."""
    result: list[HumanMessage | AIMessage] = []
    for msg in messages:
        t = _msg_type(msg)
        content = _msg_content(msg)
        if not content.strip():
            continue
        if t in ("human", "user"):
            result.append(HumanMessage(content=content))
        elif t in ("ai", "assistant"):
            result.append(AIMessage(content=content))
    return result[-6:]


_is_eligibility_inquiry = is_eligibility_inquiry


##############################################################
# 도구 정의 (2개 — LLM이 호출 가능한 도구만)
# _search_web은 run()에서 직접 제어하므로 _TOOLS에서 제외
##############################################################

_PROFILE_FIELDS = {
    "birth_date":  "나이 또는 생년월일",
    "zipcd":       "거주 지역 (시/도)",
    "jobcd":       "취업 상태 (미취업/재직/자영업 등)",
    "mrgsttscd":   "혼인 여부 (미혼/기혼/이혼)",
    "schoolcd":    "학력 (고졸/대학재학/대학졸업 등)",
    "sbizcd":      "특수 분류 (장애/한부모/다문화 등)",
}


async def extract_user_profile(
    conversation_text: str,
    existing_profile_json: str = "{}",
) -> tuple[str, dict]:
    """
    대화 내용에서 사용자 정보를 추출하고, 비어있는 항목에 대해서만 유도 질문을 생성합니다.
    role='user'일 때 run()에서 직접 호출합니다 (LLM 도구 아님).

    Args:
        conversation_text: 사용자와의 대화 내용
        existing_profile_json: 기존에 수집된 사용자 프로필 JSON 문자열
    Returns:
        tuple[str, dict]: (유도 질문 텍스트, 새로 파악된 프로필 필드 dict)
    """
    logger.info("extract_user_profile 실행")
    try:
        existing: dict = json.loads(existing_profile_json) if existing_profile_json else {}
    except (json.JSONDecodeError, TypeError):
        existing = {}

    filled = {k: v for k, v in existing.items() if v is not None and v != "" and v != "제한없음"}
    empty_fields = {k: label for k, label in _PROFILE_FIELDS.items() if not filled.get(k)}

    filled_summary = (
        "\n".join(f"  - {_PROFILE_FIELDS.get(k, k)}: {v}" for k, v in filled.items() if k in _PROFILE_FIELDS)
        or "없음"
    )
    empty_summary = (
        "\n".join(f"  - {k} ({label})" for k, label in empty_fields.items())
        or "없음 (모든 항목 수집 완료)"
    )

    try:
        result = await _profile_extractor.ainvoke([
            SystemMessage(content=(
                "사용자와의 대화에서 새로 파악된 정보를 추출하고, 아래 JSON 형식으로만 응답하세요.\n\n"
                f"이미 알고 있는 정보 (new_fields에 포함하지 마세요):\n{filled_summary}\n\n"
                f"아직 모르는 항목 (이 중에서만 추출하고 질문하세요):\n{empty_summary}\n\n"
                "응답 형식 (JSON만, 다른 텍스트 금지):\n"
                '{\n'
                '  "new_fields": {"필드명": "값", ...},\n'
                '  "questions": "아직 모르는 항목이 있으면 반드시 1~2가지 유도 질문을 생성하세요. 아직 모르는 항목이 없을 때만 빈 문자열."\n'
                '}\n\n'
                "new_fields 규칙:\n"
                "- 대화에서 명확히 언급된 값만 포함\n"
                "- 언급되지 않은 항목은 포함하지 마세요\n"
                "- 필드명은 반드시 아직 모르는 항목의 키(예: jobcd)를 사용하세요"
            )),
            HumanMessage(content=conversation_text),
        ])
        raw = _strip_code_fence(str(result.content))
        parsed = json.loads(raw)
        new_fields: dict = {k: v for k, v in parsed.get("new_fields", {}).items() if v and k in _PROFILE_FIELDS}
        questions: str = parsed.get("questions", "")
    except (json.JSONDecodeError, AttributeError):
        logger.warning("extract_user_profile JSON 파싱 실패")
        new_fields = {}
        questions = ""
    except Exception as e:
        logger.error("extract_user_profile 실패: %s", e)
        new_fields = {}
        questions = ""

    return questions, new_fields


async def _search_web(query: str, exclude_titles: list[str], count: int) -> str:
    """run()에서 직접 호출하는 웹 검색 함수 (LLM 도구 아님)."""
    try:
        tavily = TavilySearchResults(max_results=count * 2, search_depth="basic")
        results = await tavily.ainvoke({"query": query})
        exclude_set = set(exclude_titles)
        collected = []
        for r in results:
            if len(collected) >= count:
                break
            title = r.get("title", "")
            if title in exclude_set:
                continue
            collected.append(
                f"[{len(collected) + 1}] {title}\n"
                f"출처: {r.get('url', '')}\n"
                f"요약: {r.get('content', '')}"
            )
        if not collected:
            return ""
        logger.info("웹 검색 완료: %d건", len(collected))
        return "\n\n".join(collected)
    except Exception as e:
        logger.error("웹 검색 실패: %s", e)
        return ""


##############################################################
# 시스템 프롬프트
##############################################################

_SYSTEM_PROMPT = """당신은 청년 복지문화 정책 전문가입니다.
제공된 [정책 정보]에만 근거하여 답변하세요. 정보에 없는 정책이나 수치를 임의로 만들어내지 마세요.

## 도구 사용 규칙

### calculate_dday (마감일 관련 질문 시)
- 사용자가 마감일을 물을 때 호출하세요.
- deadline: bizPrdEndYmd 값, apply_period_type: aplyPrdSeCd 값을 사용하세요.
- [정책 구조화 데이터]의 dday 필드 표시 규칙 (반드시 준수):
  - "마감" → "신청 마감"으로 표시
  - "상시접수" → "상시접수" 그대로 표시
  - "D-숫자" (예: D-30) → 그대로 표시
  - "오늘 마감" → 그대로 표시
  - "정보 없음" → 마감일 항목 생략
  - **절대 "D-마감", "D-상시접수" 등 dday 값을 D- 뒤에 붙이지 마세요.**

## 답변 형식 (인사말·맺음말 없이 곧바로 아래 본문부터)

### [정책명]
- 개요: 정책 설명 + 지원 규모/대상
- 지원 내용: 지원 금액, 기간, 방식 등 구체적 내용
- 참여 자격: 연령/소득/취업/지역/학력 등 조건
- 신청 방법: 신청 기간, 경로, 절차 요약
- 신청 URL: 제공된 URL (없으면 "정보 없음")

(정책이 여러 개면 위 ### 블록을 정책 수만큼 반복)

[필수 확인사항]
✓ 신청 마감일 반드시 확인
✓ 신청 자격 요건 사전 점검
✓ 필요 서류 사전 준비

## 사용자 프로필 안내 지침
- [사용자 프로필]에 정보가 있으면 해당 조건에 맞는 정책을 우선 안내하세요.
- 프로필이 없거나 비로그인(guest)이면 일반적인 복지 정책을 안내하세요.

## 데이터 사용 규칙
- [정책 구조화 데이터]는 도구 호출 참고용입니다. JSON 형식 그대로 답변에 출력하지 마세요.
- dday, plcyNo 등 내부 필드명·값을 답변에 그대로 노출하지 마세요."""


##############################################################
# Agent 클래스 정의
##############################################################

# 자격 판정 도구는 더 이상 에이전트에 노출하지 않는다(자격 검증은 supervisor 구조화 경로 전담).
# 남는 공통 도구는 마감일 계산뿐이라 role 구분 없이 동일하다.
_TOOLS = [
    calculate_dday,
]


class WelfareAgent:
    """
    복지문화 Agent — 생성(generation) 전용.

    정책 검색은 welfare_search_node가 담당하고 state에 기록한다.
    이 에이전트는:
    1. [사용자 프로필]을 system prompt에 주입해 맞춤 답변 생성
    2. role='user'일 때 extract_user_profile 도구로 빈 항목만 추가 수집
    3. RAG 결과 부족 시 run()에서 직접 웹 검색 보완 (_search_web)
    """

    def __init__(self, model: str = "openai:gpt-4o-mini") -> None:
        self.logger = logging.getLogger(f"{__name__}.WelfareAgent")
        self._model = model

    def _make_agent(self, user_profile: dict, user_role: str = "guest"):
        profile_context = _build_profile_context(user_profile)
        system_prompt = _SYSTEM_PROMPT + f"\n\n현재 사용자 role: {user_role}"
        if profile_context:
            system_prompt += f"\n\n{profile_context}"
        system_prompt += OUTPUT_FORMAT_GUIDE
        return create_agent(model=self._model, tools=_TOOLS, system_prompt=system_prompt)

    async def run(
        self,
        inquiry: str,
        knowledge: str,
        policies: list[dict] | None = None,
        user_role: str = "guest",
        user_profile: dict | None = None,
        inquiry_type: str = "검색",
        messages: list | None = None,
    ) -> tuple[str, list[dict], Literal["rag", "web", "none"], dict, list[str]]:
        user_profile = user_profile or {}
        policies = policies or []
        web_searched = False

        # 웹 보완 임계값. 상세조회는 특정 정책 1개에 집중하므로(검색 k도 작음),
        # RAG가 1건이라도 있으면 보완하지 않는다(불필요한 웹 패딩 방지).
        # 그 외(검색·추천·비교)는 기존대로 5건 미만이면 웹으로 보완.
        min_rag = 1 if inquiry_type == "상세조회" else _MIN_RAG_COUNT

        if not knowledge and not policies:
            # RAG 완전 실패 — 웹 검색으로 대체 (상세조회도 0건이면 웹에서 그 정책을 찾는다)
            logger.info("RAG 결과 없음 — 웹 검색으로 대체")
            web_result = await _search_web(inquiry, [], 5)
            if web_result:
                knowledge = f"[웹 검색 결과]\n{web_result}"
                web_searched = True

        elif len(policies) < min_rag:
            # RAG 부족 — 웹으로 보완 (RAG 결과는 앞에 유지)
            logger.info("RAG 결과 부족(%d건) — 웹 검색으로 보완", len(policies))
            existing_titles = [p.get("plcyNm", "") for p in policies]
            web_result = await _search_web(inquiry, existing_titles, min_rag - len(policies))
            if web_result:
                knowledge += f"\n\n[웹 검색 보완]\n{web_result}"
                logger.info("웹 보완 완료")

        # 모든 RAG 정책에 dday 주입
        for policy in policies:
            policy["dday"] = calculate_dday.invoke({
                "deadline": policy.get("bizPrdEndYmd") or "",
                "apply_period_type": policy.get("aplyPrdSeCd") or "",
            })

        # role='user'일 때 대화에서 새 프로필 정보 추출 후 merge
        if user_role == "user":
            _, new_fields = await extract_user_profile(
                conversation_text=inquiry,
                existing_profile_json=json.dumps(user_profile, ensure_ascii=False),
            )
            if new_fields:
                user_profile = {**user_profile, **new_fields}
                logger.info("user_profile 업데이트: %s", list(new_fields.keys()))

        # 웹 전체 대체 시 구조화 데이터 없음 (형식 불일치) — RAG 정책만 전달
        rag_policies = [] if web_searched else policies

        # 자격 검증(build_eligibility_report) 첨부는 welfare_node가 담당한다.
        # 이번 턴 검색 정책뿐 아니라 후속질문(resolved_policies) 경로까지 붙여야 하는데,
        # resolved_policies는 state에만 있고 이 run()에는 없기 때문이다.

        agent = self._make_agent(user_profile, user_role)
        prompt = (
            "다음 정보를 바탕으로 복지문화 정책 답변을 작성하세요.\n\n"
            f"[질문]\n{inquiry}\n\n"
            f"[정책 정보]\n{knowledge or '(검색된 정책 없음)'}\n"
        )
        if rag_policies:
            prompt += (
                f"\n[정책 구조화 데이터]\n"
                f"{json.dumps(rag_policies, ensure_ascii=False, indent=2)}\n"
                "마감일 도구 호출 시 위 데이터를 사용하세요."
            )

        # 비교 모드: 정형 표는 composer 담당. 에이전트는 복지 관점 코멘트만.
        if inquiry_type == "비교":
            prompt += COMPARISON_COMMENT_GUIDE
        # 상세조회: 특정 정책 1개 깊이 안내. 단, 자격 확인 질문이면 자격 판정 결과가 우선.
        elif inquiry_type == "상세조회" and not _is_eligibility_inquiry(inquiry):
            prompt += OUTPUT_DETAIL_GUIDE
        # 추천: 각 정책 블록에 추천 이유 + 적합도 점수를 노출.
        elif inquiry_type == "추천":
            prompt += RECOMMEND_GUIDE

        prompt += SUGGESTIONS_PROMPT
        history = _build_agent_history(messages or [])
        result = await agent.ainvoke({"messages": history + [HumanMessage(content=prompt)]})  # type: ignore[arg-type]
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    logger.info("도구 호출: %s | args keys: %s", tc["name"], list(tc["args"].keys()))
        text, suggestions = parse_suggestions(str(result["messages"][-1].content))

        if web_searched:
            source: Literal["rag", "web", "none"] = "web"
        elif policies:
            source = "rag"
        else:
            source = "none"

        return text, policies, source, user_profile, suggestions


WelfareAgentDep = Annotated[WelfareAgent, Depends(WelfareAgent)]
