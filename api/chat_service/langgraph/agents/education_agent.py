import json
import logging
import re
from typing import Annotated
from fastapi import Depends
from langchain.agents import create_agent
from langchain.tools import tool
from ..constants import AGENT_CATEGORY, OUTPUT_FORMAT_GUIDE

logger = logging.getLogger(__name__)

_CATEGORY = AGENT_CATEGORY["education"]

#----------------------------------------------
# 소득분위 추정 상수 — 2024년 기준 중위소득 (월, 원)
#----------------------------------------------

_MEDIAN_INCOME = {
    1: 2_228_445,
    2: 3_682_609,
    3: 4_714_657,
    4: 5_729_913,
    5: 6_695_735,
    6: 7_618_369,
    7: 8_514_994,
}

# 소득분위 구간 — (중위소득 대비 상한 비율, 분위)
_INCOME_GRADE_THRESHOLDS = [
    (0.30, 1),
    (0.50, 2),
    (0.70, 3),
    (0.90, 4),
    (1.00, 5),
    (1.30, 6),
    (1.50, 7),
    (2.00, 8),
    (2.50, 9),
]

#----------------------------------------------
# 급여/비급여 분류 키워드
#----------------------------------------------

_COVERED_KEYWORDS = [
    "내일배움카드", "국민내일배움카드", "고용보험", "훈련비 지원", "수강료 지원",
    "훈련수당", "훈련장려금", "hrd", "직업훈련", "고용노동부 지원",
]
_SELF_FUNDED_KEYWORDS = [
    "자기부담", "본인부담", "자비", "수강료 납부", "전액 자부담",
]

#----------------------------------------------
# 학점 조건 파싱 패턴
#----------------------------------------------

_GPA_PATTERN = re.compile(
    r"(?:학점|평점|성적|gpa)[^\d]{0,10}(\d+\.?\d*)\s*(?:이상|점|/)",
    re.IGNORECASE,
)

#----------------------------------------------
# 도구 정의
#----------------------------------------------

@tool
def estimate_income_grade(monthly_income: int, family_size: int) -> str:
    """
    가구 월소득과 가구원 수로 소득분위를 추정합니다.
    다음 상황에서 반드시 호출하세요:
    - 사용자가 소득분위를 모른다고 할 때
    - 사용자가 월소득과 가구원 수를 직접 제시하며 소득분위를 물을 때
    - 장학금·교육 지원 신청 가능 여부를 판단할 때
    [사용자 정보]에 월소득(earncndsecd, 단위: 원)이 있으면 monthly_income으로 바로 사용하되,
    가구원 수(family_size)를 사용자가 명시하지 않았으면 임의로 가정하지 말고 반드시 먼저 물어보세요.

    Args:
        monthly_income: 가구 월 소득 합계 (원, 예: 3500000)
        family_size: 가구원 수 (명, 예: 4)
    Returns:
        str: 추정 소득분위 및 국가장학금 신청 가능 여부
    """
    logger.info("estimate_income_grade 실행: income=%d, family_size=%d", monthly_income, family_size)
    capped = min(max(family_size, 1), 7)
    median = _MEDIAN_INCOME[capped]
    ratio = monthly_income / median

    grade = 10
    for threshold, g in _INCOME_GRADE_THRESHOLDS:
        if ratio <= threshold:
            grade = g
            break

    lines = [
        "[ 소득분위 추정 결과 ]",
        f"  가구원 수     : {family_size}인",
        f"  월 가구소득   : {monthly_income:,}원",
        f"  기준 중위소득 : {median:,}원 ({capped}인 기준, 2024년)",
        f"  중위소득 비율 : {ratio:.1%}",
        f"  추정 소득분위 : {grade}분위",
        "",
        "[ 국가장학금 신청 가능 여부 ]",
    ]
    if grade <= 8:
        lines.append(f"  ✅ {grade}분위 — 국가장학금 신청 대상 (소득 1~8분위)")
        if grade <= 3:
            lines.append("  💡 저소득층 우선 지원 대상으로 더 높은 금액을 받을 수 있습니다.")
    else:
        lines.append(f"  ❌ {grade}분위 — 국가장학금 신청 대상 외 (소득 9~10분위 해당)")

    lines.append("※ 실제 소득분위는 한국장학재단 심사 결과와 다를 수 있습니다.")
    return "\n".join(lines)


@tool
def filter_by_gpa(policies_json: str, user_gpa: float) -> str:
    """
    사용자 학점 미달로 신청 불가한 정책을 감지합니다.
    학점 조건이 명시된 정책 중 미달인 것만 반환하므로,
    반환 목록에 없는 정책은 학점 조건 없음 또는 충족으로 간주하여 추천하세요.
    사용자가 대화에서 본인 학점을 명시적으로 언급했을 때만 호출하세요.
    학점 정보가 없으면 절대 호출하지 마세요.

    Args:
        policies_json: 정책 목록 JSON 문자열 (프롬프트의 [정책 구조화 데이터]에서 추출)
        user_gpa: 사용자 학점 (예: 3.5)
    Returns:
        str: 학점 미달로 추천에서 제외해야 할 정책 목록 (없으면 전부 추천 가능)
    """
    logger.info("filter_by_gpa 실행: user_gpa=%.2f", user_gpa)
    if user_gpa <= 0:
        return "학점 정보 없음 — 사용자가 학점을 명시적으로 언급한 경우에만 이 도구를 호출하세요."
    try:
        policies = json.loads(policies_json)
        if not isinstance(policies, list):
            raise ValueError("배열이 아님")
    except (json.JSONDecodeError, ValueError, TypeError):
        return "정책 데이터 파싱 실패 — policies_json을 올바른 JSON 배열로 전달하세요."

    blocked = []

    for p in policies:
        name = p.get("plcyNm", "정책명 미상")
        content = " ".join([
            str(p.get("plcyExplnCn") or ""),
            str(p.get("ptcpPrpTrgtCn") or ""),
            str(p.get("plcySprtCn") or ""),
            str(p.get("addAplyQlfcCndCn") or ""),
        ])
        match = _GPA_PATTERN.search(content)
        if match:
            required = float(match.group(1))
            if required > 10:
                # 100점 만점 표기로 추정 — 4.5 만점 학점과 비교 불가, 스킵
                continue
            if user_gpa < required:
                blocked.append(f"  ❌ {name} (최소 {required:.1f}점 → 현재 {user_gpa:.2f}점으로 미달)")

    lines = [f"[ 학점 필터 결과 — 내 학점: {user_gpa:.2f} ]"]
    if blocked:
        lines.append(f"\n추천 제외 대상 — 학점 미달 ({len(blocked)}건):")
        lines.extend(blocked)
        lines.append("\n※ 위 정책은 추천 목록에서 제외하세요.")
    else:
        lines.append("\n학점 미달 정책 없음 — 검색된 모든 정책을 추천할 수 있습니다.")

    lines.append("※ 학점 조건이 명시되지 않은 정책은 신청 기관에 직접 확인이 필요합니다.")
    return "\n".join(lines)


@tool
def classify_training_coverage(policy_name: str, policy_content: str) -> str:
    """
    교육·훈련 정책이 급여 훈련(고용보험 지원)인지 비급여 훈련(자비 부담)인지 분류합니다.
    사용자가 훈련비 지원 여부, 내일배움카드 사용 가능 여부를 물을 때 호출하세요.

    Args:
        policy_name: 정책명
        policy_content: 정책 설명 또는 지원 내용 텍스트
    Returns:
        str: 급여/비급여/혼합/확인필요 분류 결과
    """
    logger.info("classify_training_coverage 실행: policy=%s", policy_name)
    text = f"{policy_name} {policy_content}".lower()

    covered_hits = [kw for kw in _COVERED_KEYWORDS if kw in text]
    self_hits = [kw for kw in _SELF_FUNDED_KEYWORDS if kw in text]

    lines = [f"[ {policy_name} ] 훈련비 지원 유형 분류"]

    if covered_hits and not self_hits:
        lines.append("  분류: 급여 훈련 (고용보험 지원)")
        lines.append(f"  근거 키워드: {', '.join(covered_hits)}")
        lines.append("  💡 내일배움카드 등 고용보험 기금으로 훈련비를 지원받을 수 있습니다.")
    elif self_hits and not covered_hits:
        lines.append("  분류: 비급여 훈련 (자비 부담)")
        lines.append(f"  근거 키워드: {', '.join(self_hits)}")
        lines.append("  💡 수강료를 본인이 전액 부담해야 합니다.")
    elif covered_hits and self_hits:
        lines.append("  분류: 혼합 (일부 지원 가능)")
        lines.append(f"  지원 근거: {', '.join(covered_hits)}")
        lines.append(f"  자부담 근거: {', '.join(self_hits)}")
        lines.append("  💡 훈련 유형에 따라 지원 여부가 다를 수 있습니다. 해당 기관에 직접 확인하세요.")
    else:
        lines.append("  분류: 확인 필요")
        lines.append("  💡 훈련비 지원 여부를 텍스트에서 확인할 수 없습니다. 해당 기관에 문의하세요.")

    return "\n".join(lines)


_TOOLS = [estimate_income_grade, filter_by_gpa, classify_training_coverage]

#----------------------------------------------
# inquiry_type별 system_prompt
#----------------------------------------------

_SYSTEM_PROMPTS = {
    "검색": """당신은 청년 교육 정책 전문가입니다.
제공된 [정책 정보]에만 근거하여 답변하세요. 정보에 없는 정책이나 수치를 임의로 만들어내지 마세요.
[정책 정보]가 비어 있으면, 관련 정책을 찾지 못했다고 솔직하게 안내하세요.

다음 도구를 적극 활용하세요:
- estimate_income_grade: 소득분위 추정이 필요할 때 호출. [사용자 정보]에 월소득(earncndsecd)이 있으면 그 값을 monthly_income으로 사용하고, 가구원 수(family_size)는 사용자에게 물어보세요
- filter_by_gpa: 사용자가 학점을 언급할 때 호출 — 학점 미달로 추천 불가한 정책을 감지하여 제외하세요. 반환 목록에 없는 정책은 그대로 추천하세요
- classify_training_coverage: 훈련비 지원 여부(급여/비급여) 분류 요청 시

답변 형식:

🎓 교육정책팀 답변:
안녕하세요, 청년 교육정책팀입니다.

[정책명 및 개요]
- 정책 설명
- 지원 규모 및 대상

[지원 내용]
- 지원 유형별 세부 내용
- 지원 금액/범위

[참여 대상 자격]
- 기본 자격 요건
- 성적/소득 조건
- 전공/학과 제한 여부

[신청 방법 및 절차]
1. 신청 기간 및 경로 확인
2. 필수 서류 준비
3. 대학(기관)을 통한 추천 또는 직접 신청
4. 자격 심사 및 최종 선정

[필수 확인사항]
✓ 신청 마감일 반드시 확인
✓ 지원 대상 학교/학과 확인
✓ 필요 서류 사전 준비

친절하게 안내해드리겠습니다. 감사합니다.""",

    "추천": """당신은 청년 교육 정책 전문가입니다.
제공된 [정책 정보]에만 근거하여 답변하세요. 정보에 없는 정책이나 수치를 임의로 만들어내지 마세요.
[정책 정보]가 비어 있으면, 관련 정책을 찾지 못했다고 솔직하게 안내하세요.

[사용자 정보]가 있으면 반드시 참고하여 해당 사용자에게 가장 잘 맞는 정책을 우선 안내하세요.
각 정책마다 왜 이 사용자에게 적합한지 추천 이유를 한 줄로 명시하세요.
[사용자 정보]가 없으면(비로그인) 일반적인 추천 기준으로 안내하세요.

다음 도구를 적극 활용하세요:
- estimate_income_grade: 소득분위 추정이 필요할 때 호출. [사용자 정보]에 월소득(earncndsecd)이 있으면 그 값을 monthly_income으로 사용하고, 가구원 수(family_size)는 사용자에게 물어보세요
- filter_by_gpa: 사용자 학점으로 신청 가능한 정책만 필터링하여 추천
- classify_training_coverage: 훈련비 지원 여부를 확인해 추천 근거로 활용

답변 형식:

🎓 교육정책팀 맞춤 추천:
안녕하세요! 입력하신 정보를 바탕으로 맞춤 교육 정책을 추천해드립니다.

[추천 정책 1] 정책명
- 추천 이유: (사용자 상황과 연결된 이유)
- 지원 내용 요약
- 신청 URL

[추천 정책 2] 정책명
...

[꼭 확인하세요]
✓ 자격 요건 최종 확인 후 신청하세요
✓ 마감일을 놓치지 마세요

감사합니다.""",

    "상세조회": """당신은 청년 교육 정책 전문가입니다.
제공된 [정책 정보]에만 근거하여 답변하세요. 정보에 없는 정책이나 수치를 임의로 만들어내지 마세요.
[정책 정보]가 비어 있으면, 관련 정책을 찾지 못했다고 솔직하게 안내하세요.

하나의 정책에 대해 항목별로 최대한 상세하게 설명하세요.
신청 절차는 단계별로 풀어서 안내하세요.

다음 도구를 적극 활용하세요:
- classify_training_coverage: 해당 정책이 급여/비급여 훈련인지 명시
- estimate_income_grade: 소득 조건이 있을 때 사용자 소득분위 추정
- filter_by_gpa: 학점 조건 확인 요청 시

답변 형식:

🎓 교육정책팀 상세 안내:
안녕하세요, 요청하신 정책을 상세히 안내해드립니다.

[정책 개요]
- 정책명 / 운영 기관
- 정책 목적
- 훈련 유형: 급여/비급여 (도구 활용)

[지원 내용 상세]
- 지원 금액 및 방식
- 지원 기간

[신청 자격 상세]
- 연령 / 학력 조건
- 소득 / 성적 기준
- 제외 대상

[신청 절차 단계별 안내]
1단계: 공고 확인 및 자격 검토
2단계: 필요 서류 준비
3단계: 신청 경로 및 방법
4단계: 심사 및 결과 발표
5단계: 지원금 수령 방법

신청 URL: (URL)
감사합니다.""",

    "비교": """당신은 청년 교육 정책 전문가입니다.
제공된 [정책 정보]에만 근거하여 답변하세요. 정보에 없는 정책이나 수치를 임의로 만들어내지 마세요.
[정책 정보]가 비어 있으면, 관련 정책을 찾지 못했다고 솔직하게 안내하세요.

검색된 정책들을 마크다운 표로 나란히 비교하고, 표 아래에 핵심 차이점을 3줄 이내로 요약하세요.

다음 도구를 적극 활용하세요:
- classify_training_coverage: 각 정책의 급여/비급여 여부를 비교 항목에 포함
- filter_by_gpa: 학점 조건이 있으면 비교 표에 포함
- estimate_income_grade: 소득 조건 비교 시 사용자 소득분위 추정

답변 형식:

🎓 교육정책팀 비교 안내:
안녕하세요! 요청하신 정책들을 비교해드립니다.

| 항목 | 정책명A | 정책명B | 정책명C |
|------|---------|---------|---------|
| 지원 금액 | | | |
| 지원 대상 | | | |
| 신청 기간 | | | |
| 소득 조건 | | | |
| 훈련 유형 | 급여/비급여 | | |
| 학점 조건 | | | |
| 신청 URL | | | |

[핵심 차이 요약]
- 차이점 1
- 차이점 2
- 차이점 3

감사합니다.""",
}


#----------------------------------------------
# 에이전트
#----------------------------------------------

class EducationAgent:
    """
    교육 Agent — 생성(generation) 전용.

    정책 검색은 education_search_node가 담당하고 state에 기록한다.
    이 에이전트는 검색된 정책 텍스트(knowledge)를 받아 답변만 생성한다.
    inquiry_type별로 미리 생성된 agent를 선택해 포맷을 분기한다.

    보유 도구:
    - estimate_income_grade: 소득분위 추정 (2024 기준 중위소득)
    - filter_by_gpa: 학점 기준 장학금 필터
    - classify_training_coverage: 급여/비급여 훈련 분류
    """

    def __init__(self, model: str = "openai:gpt-4o-mini") -> None:
        self.logger = logging.getLogger(f"{__name__}.EducationAgent")
        # inquiry_type별 agent를 초기화 시점에 미리 생성 (요청마다 생성하지 않음)
        # 모든 프롬프트 끝에 OUTPUT_FORMAT_GUIDE를 붙여 분야간 출력 형식을 통일.
        self._agents = {
            inquiry_type: create_agent(
                model=model, tools=_TOOLS, system_prompt=prompt + OUTPUT_FORMAT_GUIDE
            )
            for inquiry_type, prompt in _SYSTEM_PROMPTS.items()
        }

    async def run(
        self,
        inquiry: str,
        knowledge: str,
        policies: list | None = None,
        user_profile: dict | None = None,
        inquiry_type: str = "검색",
    ) -> str:
        """검색된 정책(knowledge)으로 답변을 생성한다. (검색은 하지 않음)

        Args:
            inquiry: 사용자 질문
            knowledge: education_search_node가 검색한 정책 텍스트
            policies: 정책 메타데이터 리스트 (도구가 filter_by_gpa에서 활용)
            user_profile: 유저 프로필 (guest면 None/빈 dict)
            inquiry_type: 질문 의도 — 검색/추천/상세조회/비교
        Returns:
            str: 생성된 사용자용 답변 텍스트
        """
        agent = self._agents.get(inquiry_type, self._agents["검색"])

        prompt = (
            "다음 정보를 바탕으로 교육 정책 답변을 작성하세요.\n\n"
            f"[질문]\n{inquiry}\n\n"
            f"[정책 정보]\n{knowledge or '(검색된 정책 없음)'}\n"
        )
        if policies:
            prompt += f"\n[정책 구조화 데이터 — filter_by_gpa, classify_training_coverage 도구에 전달 가능]\n{json.dumps(policies, ensure_ascii=False)}\n"
        if user_profile:
            prompt += f"\n[사용자 정보]\n{user_profile}\n"

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )
        return result["messages"][-1].content


EducationAgentDep = Annotated[EducationAgent, Depends(EducationAgent)]
