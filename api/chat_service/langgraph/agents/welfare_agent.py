import json
import logging
from datetime import date
from typing import Annotated, Literal
from fastapi import Depends
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from ..state import DomainResult
from ..constants import AGENT_CATEGORY

logger = logging.getLogger(__name__)

_CATEGORY = AGENT_CATEGORY["welfare"]

_EXTRACT_SYSTEM_PROMPT = """대화에서 사용자가 직접 알려준 프로필 정보만 JSON으로 추출하세요.
추출할 필드 (언급이 없거나 모른다고 하면 반드시 null):
- jobcd: 취업상태 ("미취업" / "재직" / "자영업", 언급 없으면 null)
- earncndsecd: 소득분위 (1~10 정수, 분위 불명확하면 null)
- zipcd: 거주지역 (시/도 이름, 언급 없으면 null)
- mrgsttscd: 혼인여부 ("미혼" / "기혼" / "이혼", 언급 없으면 null)
- schoolcd: 학력 ("고졸" / "대학재학" / "대학졸업" 등, 언급 없으면 null)
- sbizcd: 특수분류 ("장애" / "한부모" / "다문화" / "해당없음", 언급 없으면 null)
JSON만 반환하세요. 다른 텍스트 없이."""


##############################################################
# 유틸
##############################################################

def _calc_age(birth_date_str: str) -> int | None:
    """'YYYY-MM-DD' 형식 생년월일 → 만 나이 계산."""
    try:
        bd = date.fromisoformat(birth_date_str)
        today = date.today()
        return today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day))
    except (ValueError, TypeError):
        return None


async def _extract_profile_from_message(inquiry: str, messages: list) -> dict:
    """현재 질문과 최근 대화에서 사용자가 직접 알려준 프로필 정보를 LLM으로 추출.

    추출된 값은 DB 프로필을 덮어쓴다 (사용자 발화 우선).
    """
    def _content(m) -> str:
        if isinstance(m, dict):
            return m.get("content", "")
        return getattr(m, "content", "")

    def _role(m) -> str:
        if isinstance(m, dict):
            return "사용자" if m.get("role") == "user" else "AI"
        return "사용자" if getattr(m, "type", "") == "human" else "AI"

    recent = messages[-6:] if messages else []
    conversation = "\n".join(f"{_role(m)}: {_content(m)}" for m in recent)
    conversation += f"\n사용자: {inquiry}"

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


def _build_profile_context(user_profile: dict) -> str:
    """system prompt에 삽입할 사용자 프로필 컨텍스트 문자열을 생성한다."""
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
# 툴 정의
##############################################################

@tool(return_direct=True)
async def compare_policies(policy_list: list[dict]) -> str:
    """
    여러 정책을 한눈에 비교할 수 있는 표 형태로 변환합니다.
    사용자가 "비교해줘"라고 하면 호출하세요. 검색 결과가 2개 이상일 때 사용하세요.

    Args:
        policy_list: 비교할 정책 딕셔너리 목록
    Returns:
        str: 정책명, 지원 대상, 혜택, 마감일, 신청 URL을 포함한 비교표 텍스트
    """
    logger.info("compare_policies 실행: 정책 수=%d", len(policy_list))
    lines = ["=" * 60, "📋 정책 비교표", "=" * 60]
    for idx, policy in enumerate(policy_list, 1):
        lines.append(f"\n[정책 {idx}]")
        lines.append(f"  정책명    : {policy.get('plcyNm', '정보 없음')}")
        lines.append(f"  지원 대상 : {policy.get('addAplyQlfcCndCn', '정보 없음')}")
        lines.append(f"  지원 내용 : {policy.get('plcySprtCn', '정보 없음')}")
        lines.append(f"  마감일    : {policy.get('dday', '정보 없음')}")
        apply_url = policy.get("aplyUrlAddr", "")
        lines.append(f"  신청 URL  : {apply_url if apply_url else '정보 없음'}")
        lines.append("-" * 60)
    return "\n".join(lines)


@tool
async def policy_priority_score(user_profile: dict, policy_list: list[dict]) -> str:
    """
    사용자 정보와 정책 조건을 비교해 각 정책의 적합도 점수를 계산하고 순위를 매깁니다.
    사용자가 "추천해줘"라고 하면 호출하세요.

    Args:
        user_profile: 사용자 정보 딕셔너리
        policy_list: 점수를 매길 정책 목록
    Returns:
        str: 정책별 점수와 추천 순위 텍스트
    """
    logger.info("policy_priority_score 실행: 정책 수=%d", len(policy_list))
    user_age = user_profile.get("age") or _calc_age(str(user_profile.get("birth_date", "")))
    scored = []

    for policy in policy_list:
        score = 0
        reasons = []

        # 나이 조건 (30점)
        min_age = int(policy.get("sprtTrgtMinAge") or 0)
        max_age = int(policy.get("sprtTrgtMaxAge") or 0)
        age_ok = True
        if user_age is not None:
            if min_age and min_age > int(user_age):
                age_ok = False
            if max_age and max_age < int(user_age):
                age_ok = False
        if age_ok:
            score += 30
            reasons.append("나이 조건 충족(+30)")
        else:
            reasons.append("나이 조건 불일치(+0)")

        # 마감 임박도 (40점)
        dday = policy.get("dday", "")
        if dday == "상시접수":
            score += 20
            reasons.append("상시접수(+20)")
        elif dday in ("마감", ""):
            reasons.append("마감(+0)")
        elif dday == "오늘 마감":
            score += 40
            reasons.append("오늘 마감(+40)")
        else:
            try:
                diff = int(dday.replace("D-", ""))
                pts = 40 if diff <= 7 else 25 if diff <= 30 else 10
                score += pts
                reasons.append(f"마감 {dday}(+{pts})")
            except ValueError:
                score += 10
                reasons.append("마감일 정보 없음(+10)")

        # 취업 상태 조건 (30점)
        user_job = user_profile.get("jobcd", "")
        policy_job = policy.get("jobCd", "제한없음")
        if policy_job in ("제한없음", "") or not policy_job:
            score += 30
            reasons.append("취업 조건 제한없음(+30)")
        elif user_job and user_job == policy_job:
            score += 30
            reasons.append("취업 조건 매칭(+30)")
        else:
            reasons.append("취업 조건 불일치(+0)")

        scored.append({"title": policy.get("plcyNm", "정책명 없음"), "score": score, "reasons": reasons})

    scored.sort(key=lambda x: x["score"], reverse=True)
    lines = ["=" * 60, "🏆 정책 추천 순위 (적합도 점수 기준)", "=" * 60]
    for rank, item in enumerate(scored, 1):
        lines.append(f"\n{rank}위. {item['title']} — {item['score']}점")
        lines.append(f"   근거: {', '.join(item['reasons'])}")
    lines.append("=" * 60)
    return "\n".join(lines)


@tool
def calculate_dday(deadline: str, apply_period_type: str) -> str:
    """
    정책 신청 마감일까지 남은 일수를 계산합니다.
    마감일 관련 질문 시 호출하세요.

    Args:
        deadline: 신청 마감일 (YYYYMMDD 형식)
        apply_period_type: 신청기간 구분 ("특정기간" / "상시" / "마감")
    Returns:
        str: D-day 문자열
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
    elif diff == 0:
        return "오늘 마감"
    return f"D-{diff}"


@tool
def check_eligibility(user_profile: dict, policy_metadata: dict) -> str:
    """
    사용자 정보와 복지 정책 자격 조건을 비교해 신청 가능 여부를 판별합니다.
    사용자가 "신청 가능해?", "자격 되나?" 등을 물을 때 호출하세요.

    Args:
        user_profile: 사용자 프로필 딕셔너리
        policy_metadata: 대상 정책의 메타데이터 딕셔너리
    Returns:
        str: 판정 결과 ("가능" / "불가" / "확인필요") + 항목별 사유
    """
    policy_name = policy_metadata.get("plcyNm", "해당 정책")
    logger.info("check_eligibility 실행: policy=%s", policy_name)

    age = user_profile.get("age") or _calc_age(str(user_profile.get("birth_date", "")))

    def check_age():
        min_age = int(policy_metadata.get("sprtTrgtMinAge") or 0)
        max_age = int(policy_metadata.get("sprtTrgtMaxAge") or 0)
        if age is None:
            return True, "나이 정보 없음 — 확인 필요"
        if min_age and age < min_age:
            return False, f"나이 미달 (최소 {min_age}세, 현재 {age}세)"
        if max_age and age > max_age:
            return False, f"나이 초과 (최대 {max_age}세, 현재 {age}세)"
        return True, f"나이 조건 충족 ({age}세)"

    def check_job():
        policy_job = (policy_metadata.get("jobCd") or "제한없음").strip()
        user_job = user_profile.get("jobcd")
        if not policy_job or policy_job == "제한없음":
            return True, "취업 조건 제한없음"
        if not user_job:
            return True, "취업 상태 정보 없음 — 확인 필요"
        if user_job == policy_job:
            return True, f"취업 조건 충족 ({user_job})"
        return False, f"취업 조건 불일치 (필요: {policy_job}, 현재: {user_job})"

    def check_income():
        max_income_str = policy_metadata.get("srhmhldIncmCd") or ""
        if not max_income_str:
            return True, "소득 조건 제한없음"
        try:
            max_income = int(max_income_str)
        except ValueError:
            return True, "소득 조건 확인 불가"
        user_income = user_profile.get("earncndsecd")
        if user_income is None:
            return True, "소득 정보 없음 — 확인 필요"
        if int(user_income) <= max_income:
            return True, f"소득 조건 충족 ({user_income}분위 ≤ {max_income}분위)"
        return False, f"소득 초과 ({user_income}분위 > {max_income}분위)"

    def check_region():
        policy_region = (policy_metadata.get("plcyAplyRgnCd") or "전국").strip()
        user_region = user_profile.get("zipcd")
        if policy_region in ("전국", "제한없음", ""):
            return True, "지역 제한 없음"
        if not user_region:
            return True, "거주 지역 정보 없음 — 확인 필요"
        if user_region in policy_region or policy_region in user_region:
            return True, f"지역 조건 충족 ({user_region})"
        return False, f"지역 불일치 (필요: {policy_region}, 현재: {user_region})"

    def check_school():
        policy_school = (policy_metadata.get("schoolcd") or "제한없음").strip()
        user_school = user_profile.get("schoolcd")
        if not policy_school or policy_school == "제한없음":
            return True, "학력 조건 제한없음"
        if not user_school:
            return True, "학력 정보 없음 — 확인 필요"
        if user_school == policy_school:
            return True, f"학력 조건 충족 ({user_school})"
        return False, f"학력 조건 불일치 (필요: {policy_school}, 현재: {user_school})"

    checks = [check_age(), check_job(), check_income(), check_region(), check_school()]
    failed = [(ok, r) for ok, r in checks if ok is False]
    uncertain = [(ok, r) for ok, r in checks if "확인 필요" in r or "정보 없음" in r]

    lines = [f"[ {policy_name} ] 자격 판정 결과", "-" * 50]
    for ok, reason in checks:
        icon = "✅" if "충족" in reason or "제한없음" in reason else ("❌" if ok is False else "⚠️")
        lines.append(f"  {icon} {reason}")
    lines.append("-" * 50)

    if failed:
        lines.append("판정: ❌ 불가")
        lines.append(f"사유: {', '.join(r for _, r in failed)}")
    elif uncertain:
        lines.append("판정: ⚠️ 확인필요")
        lines.append("일부 조건 정보가 부족합니다. 해당 기관에 직접 문의하세요.")
    else:
        lines.append("판정: ✅ 가능")
        lines.append("모든 조건을 충족합니다.")
    return "\n".join(lines)


@tool
async def search_web_supplement(query: str, exclude_titles: list[str], count: int) -> str:
    """
    RAG 검색 결과가 부족할 때 웹 검색으로 정책을 보완합니다.
    검색된 정책이 없거나 부족하다고 판단될 때 호출하세요.

    Args:
        query: 검색할 질문이나 키워드
        exclude_titles: 이미 안내한 정책명 목록 (중복 제외용)
        count: 추가로 가져올 정책 수
    Returns:
        str: 추가 정책 목록 텍스트
    """
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
            return "웹 검색 결과가 없습니다."
        logger.info("웹 보충 검색 완료: %d건", len(collected))
        return "\n\n".join(collected)
    except Exception as e:
        logger.error("웹 검색 실패: %s", e)
        return f"[오류] 웹 검색 중 문제 발생: {str(e)}"


##############################################################
# 시스템 프롬프트
##############################################################

_SYSTEM_PROMPT = """당신은 청년 복지문화 정책 전문가입니다.
제공된 [정책 정보]에만 근거하여 답변하세요. 정보에 없는 정책이나 수치를 임의로 만들어내지 마세요.
[정책 정보]가 비어 있으면 search_web_supplement를 호출해 추가 정보를 검색하세요.

## 도구 사용 규칙

### search_web_supplement (RAG 결과 없거나 부족할 때)
- [정책 정보]가 비어 있거나 부족하면 호출하세요.
- exclude_titles에 이미 안내한 정책명을 넣어 중복을 방지하세요.

### calculate_dday (마감일 관련 질문 시)
- 사용자가 마감일을 물을 때 호출하세요.
- deadline: bizPrdEndYmd 값, apply_period_type: aplyPrdSeCd 값을 사용하세요.
- 마감일 데이터가 없으면 호출하지 마세요.

### compare_policies (비교 요청 시)
- 사용자가 "비교해줘"라고 하면 호출하세요.
- [정책 구조화 데이터]가 있으면 그 데이터를 사용하세요.

### policy_priority_score (추천 요청 시)
- 사용자가 "추천해줘"라고 하면 호출하세요.
- [정책 구조화 데이터]가 있으면 그 데이터를 사용하세요.

### check_eligibility (자격 확인 요청 시)
- 사용자가 "신청 가능해?", "자격 되나?" 등을 물을 때만 호출하세요.
- [정책 구조화 데이터]가 있으면 해당 정책 데이터를 그대로 전달하세요.

## 사용자 프로필 안내 지침
- [사용자 프로필]에 정보가 있으면 프로필에 맞는 정책을 우선 안내하세요.
- guest(비로그인)인 경우 일반적인 복지 정책을 안내하세요.

## 유도 질문 지침
- 답변 후 [사용자 프로필]에 미입력 항목이 있으면 더 정확한 맞춤 정책 안내를 위해 자연스럽게 1~2가지만 질문하세요.
- 단, 이미 대화에서 물어본 항목은 다시 묻지 마세요."""


##############################################################
# Agent 클래스 정의
##############################################################

class WelfareAgent:
    """
    복지문화 Agent — 생성(generation) 전용.

    정책 검색은 welfare_search_node가 담당하고 state에 기록한다.
    이 에이전트는:
    1. 대화 이력에서 사용자가 직접 언급한 프로필 정보를 추출해 DB 프로필과 병합
    2. 병합 프로필을 system prompt에 주입해 맞춤 답변 생성
    3. RAG 결과 부족 시 search_web_supplement 툴로 웹 보충 검색 수행
    """

    def __init__(self, model: str = "openai:gpt-4o-mini") -> None:
        self.logger = logging.getLogger(f"{__name__}.WelfareAgent")
        self._model = model

    def _make_agent(self, user_profile: dict):
        system_prompt = _SYSTEM_PROMPT
        profile_context = _build_profile_context(user_profile)
        if profile_context:
            system_prompt = f"{system_prompt}\n\n{profile_context}"
        return create_agent(
            model=self._model,
            tools=[
                search_web_supplement,
                calculate_dday,
                compare_policies,
                policy_priority_score,
                check_eligibility,
            ],
            system_prompt=system_prompt,
        )

    async def run(
        self,
        inquiry: str,
        knowledge: str,
        policies: list[dict] | None = None,
        user_role: str = "guest",
        user_profile: dict | None = None,
        messages: list | None = None,
    ) -> DomainResult:
        """검색된 정책(knowledge, policies)으로 맞춤 답변을 생성하고 DomainResult로 반환한다.

        1. 대화 이력에서 사용자 발화 프로필 추출 → DB 프로필과 병합
        2. 병합 프로필을 system prompt에 주입
        3. 에이전트 호출 → 답변 생성 (RAG 부족 시 search_web_supplement 자동 호출)

        Args:
            inquiry: 사용자 질문
            knowledge: welfare_search_node가 검색한 정책 텍스트 (LLM 답변용)
            policies: welfare_search_node가 추출한 raw 정책 메타 (툴 호출용)
            user_role: "guest" 또는 "user"
            user_profile: DB에서 조회한 로그인 유저 프로필 (guest면 None/빈 dict)
            messages: 이전 대화 이력 (프로필 추출용)
        Returns:
            DomainResult: 생성된 답변 + 정책 메타 + 출처
        """
        user_profile = user_profile or {}
        policies = policies or []
        messages = messages or []

        # Step 1: 로그인 유저면 대화 이력에서 프로필 추출 후 DB 프로필과 병합
        merged_profile = user_profile
        if user_role == "user":
            extracted = await _extract_profile_from_message(inquiry, messages)
            if extracted:
                merged_profile = {**user_profile, **extracted}
                logger.info("프로필 병합 완료: %s", list(extracted.keys()))

        agent = self._make_agent(merged_profile)

        prompt = (
            "다음 정보를 바탕으로 복지문화 정책 답변을 작성하세요.\n\n"
            f"[질문]\n{inquiry}\n\n"
            f"[정책 정보]\n{knowledge or '(검색된 정책 없음 — search_web_supplement를 호출하세요)'}\n"
        )

        if policies:
            prompt += (
                f"\n[정책 구조화 데이터]\n"
                f"{json.dumps(policies, ensure_ascii=False, indent=2)}\n"
                "비교/추천/자격판정 도구 호출 시 위 구조화 데이터를 사용하세요."
            )

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )
        text = result["messages"][-1].content

        # source 결정: tool 호출 이력 기반
        called_tools = {
            msg.name for msg in result["messages"]
            if hasattr(msg, "name") and msg.name
        }
        if policies:
            source: Literal["rag", "web", "none"] = "rag"
        elif "search_web_supplement" in called_tools:
            source = "web"
        else:
            source = "none"

        return DomainResult(
            text=text,
            policies=policies,
            category=_CATEGORY,
            source=source,
        )


WelfareAgentDep = Annotated[WelfareAgent, Depends(WelfareAgent)]
