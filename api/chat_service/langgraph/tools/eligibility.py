"""정책 자격 진단(추천) 공통 tool — 도메인 검색 노드와 welfare 에이전트가 공유한다.

- check_policy_eligibility / eligibility_sort_key / format_verdict_line:
  LLM 미사용, 검색 노드에서 자격 적합순 정렬·라벨링에 사용 (나이/지역/혼인 공통 검사).
- check_eligibility_detailed (@tool):
  LLM이 호출하는 상세 자격 판정 툴. 나이/지역/혼인 + 취업/소득/학력까지 검사.
- user_profile이 비어있으면(=게스트) check_policy_eligibility는 None을 반환해 진단을 스킵한다.
"""
import logging
import re
from typing import Literal, TypedDict

from langchain.tools import tool

from .age import calc_age

logger = logging.getLogger(__name__)

# 자격 확인 질문 감지 — welfare_agent / welfare_search_node 공유
ELIGIBILITY_RE = re.compile(
    r"(자격|조건|대상)\s*(이|가)?\s*(돼|됩|되나요|맞아|맞나요)"
    r"|신청\s*(가능|할\s*수)"
    r"|지원\s*(가능|돼|되나요)"
    r"|받을\s*수\s*있"
    r"|나도\s*(가능|돼)"
)


def is_eligibility_inquiry(inquiry: str) -> bool:
    return bool(ELIGIBILITY_RE.search(inquiry))


# 자격 판정: 적합 / 미충족 / 판단불가
Verdict = Literal["eligible", "ineligible", "unknown"]


class EligibilityResult(TypedDict):
    verdict: Verdict
    reasons: list[str]


# 나이 제한 검사
def _check_age(meta: dict, age: int | None) -> tuple[Verdict, str]:
    # 정책이 연령 제한 없음을 명시하면 무조건 통과
    if meta.get("sprtTrgtAgeLmtYn") == "N":
        return "eligible", "연령 ✓ (제한 없음)"

    if age is None:
        return "unknown", "연령: 생년월일 정보 없음"

    lo_raw, hi_raw = meta.get("sprtTrgtMinAge"), meta.get("sprtTrgtMaxAge")
    lo = int(lo_raw) if lo_raw else None
    hi = int(hi_raw) if hi_raw else None

    # 데이터에 연령 정보가 아예 없을 경우 → unknown
    if lo is None and hi is None:
        return "unknown", "연령조건 : 정책 데이터 부족"

    # 있는 조건만 검사 → 하나라도 위배되면 미충족
    if lo is not None and age < lo:
        return "ineligible", f"연령 ✗ (만 {age}세, 최소 {lo}세 이상)"
    if hi is not None and age > hi:
        return "ineligible", f"연령 ✗ (만 {age}세, 최대 {hi}세 이하)"

    # 통과 — 메시지에 알려진 조건만 표시
    bound_msg = f"{lo or '?'}~{hi or '?'}세"
    return "eligible", f"연령 ✓ (만 {age}세, 요건 {bound_msg})"


# 지역 검사
def _check_region(meta: dict, user_region: str | None) -> tuple[Verdict, str]:
    region_code = meta.get("zipCd") or ""

    # 빈 값 → 지역 제한 없음 (전국 허용)
    if not region_code:
        return "eligible", "지역 ✓ (제한없음)"
    # 사용자가 빈 값 → unknown
    if not user_region:
        return "unknown", "지역: 사용자 거주지 정보 없음"

    # 양방향 substring — "전북" / "전라북도" / "전북특별자치도" 어떤 형태든 매칭되게
    parts = [p.strip() for p in region_code.split(",") if p.strip()]
    for p in parts:
        if p in user_region or user_region in p:
            return "eligible", f"지역 ✓ ({user_region})"
    return "ineligible", f"지역 ✗ (요건: {region_code} / 거주지: {user_region})"


# 결혼 여부 검사
def _check_marriage(meta: dict, user_mrg: str | None) -> tuple[Verdict, str]:
    pol_mrg = meta.get("mrgSttsCd")
    # 정책이 혼인 무관 → eligible
    if pol_mrg == "제한없음":
        return "eligible", "혼인 ✓ (제한없음)"
    # 데이터가 빈값 → unknown
    if not pol_mrg:
        return "unknown", "혼인 요건: 정책 데이터 부족"
    # 사용자가 혼인여부를 채우지 않은 경우
    if not user_mrg:
        return "unknown", "혼인: 사용자 정보 없음"
    # 사용자의 혼인여부와 정책의 혼인여부가 일치
    if user_mrg == pol_mrg:
        return "eligible", f"혼인 ✓ ({user_mrg})"
    # 사용자의 혼인여부가 정책 혼인제한에 걸림
    return "ineligible", f"혼인 ✗ (요건: {pol_mrg} / 현재: {user_mrg})"


def _check_job(meta: dict, user_job: str | None) -> tuple[Verdict, str]:
    policy_job = (meta.get("jobCd") or "제한없음").strip()
    if not policy_job or policy_job == "제한없음":
        return "eligible", "취업 ✓ (제한없음)"
    if not user_job:
        return "unknown", "취업: 사용자 정보 없음"
    if user_job == policy_job:
        return "eligible", f"취업 ✓ ({user_job})"
    return "ineligible", f"취업 ✗ (요건: {policy_job} / 현재: {user_job})"


def _check_income(meta: dict, user_income: str | None) -> tuple[Verdict, str]:
    max_income_str = (meta.get("srhmhldIncmCd") or "").strip()
    if not max_income_str:
        return "eligible", "소득 ✓ (제한없음)"
    if not user_income:
        return "unknown", "소득: 사용자 정보 없음"
    try:
        if int(user_income) <= int(max_income_str):
            return "eligible", f"소득 ✓ ({user_income}분위 ≤ {max_income_str}분위)"
        return "ineligible", f"소득 ✗ ({user_income}분위 > {max_income_str}분위)"
    except (ValueError, TypeError):
        return "unknown", "소득 조건 확인 불가"


def _check_school(meta: dict, user_school: str | None) -> tuple[Verdict, str]:
    policy_school = (meta.get("schoolcd") or "제한없음").strip()
    if not policy_school or policy_school == "제한없음":
        return "eligible", "학력 ✓ (제한없음)"
    if not user_school:
        return "unknown", "학력: 사용자 정보 없음"
    if user_school == policy_school:
        return "eligible", f"학력 ✓ ({user_school})"
    return "ineligible", f"학력 ✗ (요건: {policy_school} / 현재: {user_school})"


def check_policy_eligibility(
    metadata: dict, user_profile: dict | None
) -> EligibilityResult | None:
    """단일 정책 자격 진단. user_profile이 비어있으면(게스트) None."""
    if not user_profile:
        return None

    age = calc_age(user_profile.get("birth_date"))
    checks: list[tuple[Verdict, str]] = [
        _check_age(metadata, age),
        _check_region(metadata, user_profile.get("zipcd")),
        _check_marriage(metadata, user_profile.get("mrgsttscd")),
    ]

    reasons = [r for _, r in checks]
    verdicts = {v for v, _ in checks}
    if "ineligible" in verdicts:
        final: Verdict = "ineligible"
    elif "unknown" in verdicts:
        final = "unknown"
    else:
        final = "eligible"
    return EligibilityResult(verdict=final, reasons=reasons)


def format_verdict_line(result: EligibilityResult) -> str:
    """자격 진단 결과를 knowledge에 끼워 넣을 한 줄 라벨로 변환."""
    icon = {
        "eligible":   "✅ 자격 적합",
        "ineligible": "❌ 자격 미충족",
        "unknown":    "❓ 자격 확인 필요",
    }[result["verdict"]]
    return f"{icon} (나이·지역·혼인 기준) — {' | '.join(result['reasons'])}"


# 검색 결과 정렬 우선순위: eligible → unknown → ineligible
VERDICT_ORDER = {"eligible": 0, "unknown": 1, "ineligible": 2}


@tool
def check_eligibility_detailed(user_profile: dict, policy_metadata: dict) -> str:
    """
    사용자 프로필과 복지 정책 자격 조건을 교차 검증해 신청 가능 여부를 판별합니다.
    사용자가 "신청 가능해?", "자격 되나?" 등을 물을 때 호출하세요.
    RAG 검색([정책 구조화 데이터])으로 가져온 정책에만 사용하세요. 웹 검색 결과에는 사용하지 마세요.

    Args:
        user_profile: 사용자 프로필 딕셔너리 (나이·취업상태·소득·지역·학력·혼인 등)
        policy_metadata: RAG로 가져온 정책의 메타데이터 딕셔너리
    Returns:
        str: 항목별 충족 여부(✅/❌/⚠️) + 최종 판정 결과
    """
    policy_name = policy_metadata.get("plcyNm", "해당 정책")
    logger.info("check_eligibility_detailed 실행 — 정책: %s", policy_name)
    age = calc_age(user_profile.get("birth_date"))
    checks: list[tuple[Verdict, str]] = [
        _check_age(policy_metadata, age),
        _check_region(policy_metadata, user_profile.get("zipcd")),
        _check_marriage(policy_metadata, user_profile.get("mrgsttscd")),
        _check_job(policy_metadata, user_profile.get("jobcd")),
        _check_income(policy_metadata, user_profile.get("earncndsecd")),
        _check_school(policy_metadata, user_profile.get("schoolcd")),
    ]
    lines = [f"[ {policy_name} ] 자격 판정 결과", "-" * 50]
    for verdict, reason in checks:
        icon = "✅" if verdict == "eligible" else ("❌" if verdict == "ineligible" else "⚠️")
        lines.append(f"  {icon} {reason}")
    lines.append("-" * 50)

    verdicts = {v for v, _ in checks}
    if "ineligible" in verdicts:
        lines.append("판정: ❌ 불가")
        lines.append(f"사유: {', '.join(r for v, r in checks if v == 'ineligible')}")
    elif "unknown" in verdicts:
        lines.append("판정: ⚠️ 확인필요")
        lines.append("일부 조건 정보가 부족합니다. 해당 기관에 직접 문의하세요.")
    else:
        lines.append("판정: ✅ 가능")
        lines.append("모든 조건을 충족합니다.")

    return "\n".join(lines)
