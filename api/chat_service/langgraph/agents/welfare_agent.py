import json
import logging
from datetime import date
from typing import Annotated, Literal
from fastapi import Depends
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model
from ..constants import AGENT_CATEGORY

logger = logging.getLogger(__name__)

_CATEGORY = AGENT_CATEGORY["welfare"]
_MIN_RAG_COUNT = 3  # RAG 결과가 이 수 미만이면 웹 검색으로 보완

##############################################################
# 유틸
##############################################################

def _calc_age(birth_date_str: str) -> int | None:
    try:
        bd = date.fromisoformat(birth_date_str)
        today = date.today()
        return today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day))
    except (ValueError, TypeError):
        return None


def _last_ai_message(messages: list) -> str:
    for m in reversed(messages):
        role = m.get("role", "") if isinstance(m, dict) else getattr(m, "type", "")
        content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
        if role in ("ai", "assistant") and content:
            return content
    return ""


async def _normalize_web_policies(web_text: str) -> list[dict]:
    llm = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0)
    try:
        result = await llm.ainvoke([
            SystemMessage(content=(
                "웹 검색 결과에서 청년 복지 정책 정보를 추출해 JSON 배열로 반환하세요.\n"
                "각 정책은 다음 필드를 포함하세요 (없으면 null):\n"
                "plcyNm, plcyExplnCn, plcySprtCn, ptcpPrpTrgtCn, addAplyQlfcCndCn, "
                "aplyUrlAddr, aplyYmd, bizPrdEndYmd, aplyPrdSeCd\n"
                "JSON 배열만 반환하세요."
            )),
            HumanMessage(content=web_text),
        ])
        text = str(result.content).strip().strip("```json").strip("```").strip()
        policies = json.loads(text)
        if not isinstance(policies, list):
            raise ValueError("반환값이 배열이 아님")
        logger.info("웹 정책 정규화 완료: %d건", len(policies))
        return policies
    except Exception as e:
        logger.warning("웹 정책 정규화 실패: %s", e)
        return []


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


##############################################################
# 도구 정의 (3개 — LLM이 호출 가능한 도구만)
# search_web_supplement는 run()에서 직접 제어하므로 _TOOLS에서 제외
##############################################################

_PROFILE_FIELDS = {
    "birth_date":  "나이 또는 생년월일",
    "zipcd":       "거주 지역 (시/도)",
    "jobcd":       "취업 상태 (미취업/재직/자영업 등)",
    "earncndsecd": "가구 소득 분위 (1~10분위)",
    "mrgsttscd":   "혼인 여부 (미혼/기혼/이혼)",
    "schoolcd":    "학력 (고졸/대학재학/대학졸업 등)",
    "sbizcd":      "특수 분류 (장애/한부모/다문화 등)",
}


async def extract_user_profile(conversation_text: str, existing_profile_json: str = "{}") -> str:
    """
    대화 내용에서 사용자 정보를 추출하고, 비어있는 항목에 대해서만 유도 질문을 생성합니다.
    role='user'일 때 run()에서 직접 호출합니다 (LLM 도구 아님).

    Args:
        conversation_text: 사용자와의 대화 내용
        existing_profile_json: 기존에 수집된 사용자 프로필 JSON 문자열
    Returns:
        str: 이번 대화에서 새로 확인된 정보 요약 + 비어있는 항목에 대한 질문 (1~2가지)
    """
    logger.info("extract_user_profile 실행")
    try:
        existing: dict = json.loads(existing_profile_json) if existing_profile_json else {}
    except (json.JSONDecodeError, TypeError):
        existing = {}

    filled = {k: v for k, v in existing.items() if v and v not in ("", None)}
    empty_fields = {k: label for k, label in _PROFILE_FIELDS.items() if not filled.get(k)}

    filled_summary = (
        "\n".join(f"  - {_PROFILE_FIELDS.get(k, k)}: {v}" for k, v in filled.items() if k in _PROFILE_FIELDS)
        or "없음"
    )
    empty_summary = (
        "\n".join(f"  - {label}" for label in empty_fields.values())
        or "없음 (모든 항목 수집 완료)"
    )

    llm = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0)
    result = await llm.ainvoke([
        SystemMessage(content=(
            "사용자와의 대화에서 새로 파악된 정보를 추출하세요.\n"
            "그리고 아직 모르는 항목 중에서 가장 중요한 1~2가지만 자연스럽게 질문하세요.\n\n"
            f"이미 알고 있는 정보 (질문하지 마세요):\n{filled_summary}\n\n"
            f"아직 모르는 항목 (이 중에서만 질문하세요):\n{empty_summary}\n\n"
            "응답 형식:\n"
            "확인된 정보: (이번 대화에서 새로 알게 된 내용, 없으면 '없음')\n"
            "추가 질문: (비어있는 항목 중 1~2가지만, 모두 채워졌으면 생략)"
        )),
        HumanMessage(content=conversation_text),
    ])
    return str(result.content)


@tool
def check_eligibility(user_profile: dict, policy_metadata: dict) -> str:
    """
    사용자 프로필과 복지 정책 자격 조건을 교차 검증해 신청 가능 여부를 판별합니다.
    사용자가 "신청 가능해?", "자격 되나?" 등을 물을 때 호출하세요.
    RAG 검색([정책 구조화 데이터])으로 가져온 정책에만 사용하세요. 웹 검색 결과에는 사용하지 마세요.

    Args:
        user_profile: 사용자 프로필 딕셔너리 (나이·취업상태·소득·지역·학력 등)
        policy_metadata: RAG로 가져온 정책의 메타데이터 딕셔너리
    Returns:
        str: 항목별 충족 여부(✅/❌/⚠️) + 최종 판정 결과
    """
    policy_name = policy_metadata.get("plcyNm", "해당 정책")
    logger.info("check_eligibility 실행: policy=%s", policy_name)

    age = user_profile.get("age") or _calc_age(str(user_profile.get("birth_date", "")))
    checks = []

    # 나이
    min_age = int(policy_metadata.get("sprtTrgtMinAge") or 0)
    max_age = int(policy_metadata.get("sprtTrgtMaxAge") or 0)
    if age is None:
        checks.append((None, "나이 정보 없음 — 확인 필요"))
    elif min_age and age < min_age:
        checks.append((False, f"나이 미달 (최소 {min_age}세, 현재 {age}세)"))
    elif max_age and age > max_age:
        checks.append((False, f"나이 초과 (최대 {max_age}세, 현재 {age}세)"))
    else:
        checks.append((True, f"나이 조건 충족 ({age}세)" if age else "나이 조건 제한없음"))

    # 취업 상태
    policy_job = (policy_metadata.get("jobCd") or "제한없음").strip()
    user_job = user_profile.get("jobcd")
    if not policy_job or policy_job == "제한없음":
        checks.append((True, "취업 조건 제한없음"))
    elif not user_job:
        checks.append((None, "취업 상태 정보 없음 — 확인 필요"))
    elif user_job == policy_job:
        checks.append((True, f"취업 조건 충족 ({user_job})"))
    else:
        checks.append((False, f"취업 조건 불일치 (필요: {policy_job}, 현재: {user_job})"))

    # 소득
    max_income_str = policy_metadata.get("srhmhldIncmCd") or ""
    if not max_income_str:
        checks.append((True, "소득 조건 제한없음"))
    else:
        user_income = user_profile.get("earncndsecd")
        try:
            max_income = int(max_income_str)
            if user_income is None:
                checks.append((None, "소득 정보 없음 — 확인 필요"))
            elif int(user_income) <= max_income:
                checks.append((True, f"소득 조건 충족 ({user_income}분위 ≤ {max_income}분위)"))
            else:
                checks.append((False, f"소득 초과 ({user_income}분위 > {max_income}분위)"))
        except ValueError:
            checks.append((None, "소득 조건 확인 불가"))

    # 지역
    policy_region = (policy_metadata.get("plcyAplyRgnCd") or "전국").strip()
    user_region = user_profile.get("zipcd")
    if policy_region in ("전국", "제한없음", ""):
        checks.append((True, "지역 제한 없음"))
    elif not user_region:
        checks.append((None, "거주 지역 정보 없음 — 확인 필요"))
    elif user_region in policy_region or policy_region in user_region:
        checks.append((True, f"지역 조건 충족 ({user_region})"))
    else:
        checks.append((False, f"지역 불일치 (필요: {policy_region}, 현재: {user_region})"))

    # 학력
    policy_school = (policy_metadata.get("schoolcd") or "제한없음").strip()
    user_school = user_profile.get("schoolcd")
    if not policy_school or policy_school == "제한없음":
        checks.append((True, "학력 조건 제한없음"))
    elif not user_school:
        checks.append((None, "학력 정보 없음 — 확인 필요"))
    elif user_school == policy_school:
        checks.append((True, f"학력 조건 충족 ({user_school})"))
    else:
        checks.append((False, f"학력 조건 불일치 (필요: {policy_school}, 현재: {user_school})"))

    lines = [f"[ {policy_name} ] 자격 판정 결과", "-" * 50]
    for ok, reason in checks:
        icon = "✅" if ok is True else ("❌" if ok is False else "⚠️")
        lines.append(f"  {icon} {reason}")
    lines.append("-" * 50)

    failed = [r for ok, r in checks if ok is False]
    uncertain = [r for ok, r in checks if ok is None]

    if failed:
        lines.append("판정: ❌ 불가")
        lines.append(f"사유: {', '.join(failed)}")
    elif uncertain:
        lines.append("판정: ⚠️ 확인필요")
        lines.append("일부 조건 정보가 부족합니다. 해당 기관에 직접 문의하세요.")
    else:
        lines.append("판정: ✅ 가능")
        lines.append("모든 조건을 충족합니다.")

    return "\n".join(lines)


@tool
def calculate_dday(deadline: str, apply_period_type: str) -> str:
    """
    정책 신청 마감일까지 남은 일수를 계산합니다.
    사용자가 마감일을 물을 때 호출하세요.

    Args:
        deadline: 신청 마감일 (YYYYMMDD 형식, 없으면 빈 문자열)
        apply_period_type: 신청기간 구분 ("특정기간" / "상시" / "마감")
    Returns:
        str: D-day 문자열 (예: "D-30", "상시접수", "마감")
    """
    logger.info("calculate_dday 실행: deadline=%s", deadline)
    if apply_period_type == "상시":
        return "상시접수"
    if not deadline:
        return "마감"
    today = date.today()
    try:
        deadline_date = date(int(deadline[:4]), int(deadline[4:6]), int(deadline[6:8]))
    except (ValueError, IndexError):
        return "마감"
    diff = (deadline_date - today).days
    if diff < 0:
        return "마감"
    if diff == 0:
        return "오늘 마감"
    return f"D-{diff}"


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

### check_eligibility (자격 확인 요청 시)
- 사용자가 "신청 가능해?", "자격 되나?" 등을 물을 때 호출하세요.
- **[정책 구조화 데이터]에 있는 RAG 정책에만 사용하세요.** [웹 검색 보완] 결과는 데이터 형식이 맞지 않으므로 호출하지 마세요.
- user_profile에 [사용자 프로필] 값을, policy_metadata에 해당 정책 딕셔너리를 전달하세요.

### calculate_dday (마감일 관련 질문 시)
- 사용자가 마감일을 물을 때 호출하세요.
- deadline: bizPrdEndYmd 값, apply_period_type: aplyPrdSeCd 값을 사용하세요.
- [정책 구조화 데이터]의 dday 필드가 있으면 정책 안내 시 함께 표시하세요.

### 비교 요청 시 (도구 없이 직접 작성)
- 사용자가 "비교해줘"라고 하면 [정책 정보]를 바탕으로 아래 형식으로 작성하세요.

■ 정책명
  - 지원 대상:
  - 지원 내용:
  - 마감일:
  - 신청 URL:

정보가 없으면 "정보 없음"으로 표시하세요.

## 사용자 프로필 안내 지침
- [사용자 프로필]에 정보가 있으면 해당 조건에 맞는 정책을 우선 안내하세요.
- 프로필이 없거나 비로그인(guest)이면 일반적인 복지 정책을 안내하세요."""


##############################################################
# Agent 클래스 정의
##############################################################

_TOOLS = [
    check_eligibility,
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
        return create_agent(model=self._model, tools=_TOOLS, system_prompt=system_prompt)

    async def run(
        self,
        inquiry: str,
        knowledge: str,
        policies: list[dict] | None = None,
        user_role: str = "guest",
        user_profile: dict | None = None,
        messages: list | None = None,
    ) -> tuple[str, list[dict], Literal["rag", "web", "none"], dict]:
        user_profile = user_profile or {}
        policies = policies or []
        messages = messages or []
        web_searched = False

        if not knowledge and not policies:
            # RAG 완전 실패 — 이전 AI 응답 재활용 후, 없으면 웹으로 대체
            prev_ai = _last_ai_message(messages)
            if prev_ai:
                knowledge = f"[이전 답변 참고]\n{prev_ai}"
                logger.info("RAG 결과 없음 — 이전 AI 응답 fallback 사용")
            else:
                logger.info("RAG 결과 없음 — 웹 검색으로 대체")
                web_result = await _search_web(inquiry, [], 5)
                if web_result:
                    knowledge = f"[웹 검색 결과]\n{web_result}"
                    policies = await _normalize_web_policies(web_result)
                    web_searched = True

        elif len(policies) < _MIN_RAG_COUNT:
            # RAG 부족 — 웹으로 보완 (RAG 결과는 앞에 유지)
            logger.info("RAG 결과 부족(%d건) — 웹 검색으로 보완", len(policies))
            existing_titles = [p.get("plcyNm", "") for p in policies]
            web_result = await _search_web(inquiry, existing_titles, _MIN_RAG_COUNT - len(policies))
            if web_result:
                knowledge += f"\n\n[웹 검색 보완]\n{web_result}"
                logger.info("웹 보완 완료")

        # 모든 RAG 정책에 dday 주입
        for policy in policies:
            policy["dday"] = calculate_dday.invoke({
                "deadline": policy.get("bizPrdEndYmd") or "",
                "apply_period_type": policy.get("aplyPrdSeCd") or "",
            })

        # role='user'일 때 빈 프로필 필드에 대한 유도 질문 생성
        profile_questions = ""
        if user_role == "user":
            profile_questions = await extract_user_profile(
                conversation_text=inquiry,
                existing_profile_json=json.dumps(user_profile, ensure_ascii=False),
            )
            logger.info("extract_user_profile 실행 완료")

        agent = self._make_agent(user_profile, user_role)
        prompt = (
            "다음 정보를 바탕으로 복지문화 정책 답변을 작성하세요.\n\n"
            f"[질문]\n{inquiry}\n\n"
            f"[정책 정보]\n{knowledge or '(검색된 정책 없음)'}\n"
        )
        if profile_questions:
            prompt += f"\n[사용자 추가 확인 필요 항목]\n{profile_questions}\n답변 끝에 위 추가 질문을 자연스럽게 붙여주세요."
        # 웹 전체 대체 시 구조화 데이터 없음 (형식 불일치) — RAG 정책만 전달
        rag_policies = [] if web_searched else policies
        if rag_policies:
            prompt += (
                f"\n[정책 구조화 데이터]\n"
                f"{json.dumps(rag_policies, ensure_ascii=False, indent=2)}\n"
                "자격 확인·마감일 도구 호출 시 위 데이터를 사용하세요."
            )

        result = await agent.ainvoke({"messages": [{"role": "user", "content": prompt}]})
        text = str(result["messages"][-1].content)

        if web_searched:
            source: Literal["rag", "web", "none"] = "web"
        elif policies:
            source = "rag"
        else:
            source = "none"

        return text, policies, source, user_profile


WelfareAgentDep = Annotated[WelfareAgent, Depends(WelfareAgent)]
