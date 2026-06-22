import logging
from langchain_core.tools import tool
from tools.welfare.tool_user_profile import calc_age

logger = logging.getLogger(__name__)


def _check_age(user_age: int | None, policy: dict) -> tuple[bool, str]:
    min_age_str = policy.get("sprtTrgtMinAge") or "0"
    max_age_str = policy.get("sprtTrgtMaxAge") or "0"
    try:
        min_age = int(min_age_str)
        max_age = int(max_age_str)
    except ValueError:
        return True, "나이 조건 확인 불가"

    if user_age is None:
        return True, "나이 정보 없음 — 확인 필요"

    if min_age and user_age < min_age:
        return False, f"나이 미달 (최소 {min_age}세, 현재 {user_age}세)"
    if max_age and user_age > max_age:
        return False, f"나이 초과 (최대 {max_age}세, 현재 {user_age}세)"
    return True, f"나이 조건 충족 ({user_age}세)"


def _check_job(user_job: str | None, policy: dict) -> tuple[bool, str]:
    policy_job = (policy.get("jobCd") or "제한없음").strip()
    if not policy_job or policy_job == "제한없음":
        return True, "취업 조건 제한없음"
    if not user_job:
        return True, "취업 상태 정보 없음 — 확인 필요"
    if user_job == policy_job:
        return True, f"취업 조건 충족 ({user_job})"
    return False, f"취업 조건 불일치 (필요: {policy_job}, 현재: {user_job})"


def _check_income(user_income: int | None, policy: dict) -> tuple[bool, str]:
    max_income_str = policy.get("srhmhldIncmCd") or ""
    if not max_income_str:
        return True, "소득 조건 제한없음"
    try:
        max_income = int(max_income_str)
    except ValueError:
        return True, "소득 조건 확인 불가"
    if user_income is None:
        return True, "소득 정보 없음 — 확인 필요"
    if user_income <= max_income:
        return True, f"소득 조건 충족 ({user_income}분위 ≤ {max_income}분위)"
    return False, f"소득 초과 ({user_income}분위 > {max_income}분위)"


def _check_school(user_school: str | None, policy: dict) -> tuple[bool, str]:
    policy_school = (policy.get("schoolcd") or "제한없음").strip()
    if not policy_school or policy_school == "제한없음":
        return True, "학력 조건 제한없음"
    if not user_school:
        return True, "학력 정보 없음 — 확인 필요"
    if user_school == policy_school:
        return True, f"학력 조건 충족 ({user_school})"
    return False, f"학력 조건 불일치 (필요: {policy_school}, 현재: {user_school})"


def _check_sbiz(user_sbiz: str | None, policy: dict) -> tuple[bool, str]:
    policy_sbiz = (policy.get("sbizcd") or "제한없음").strip()
    if not policy_sbiz or policy_sbiz == "제한없음":
        return True, "특수 분류 제한없음"
    if not user_sbiz:
        return True, "특수 분류 정보 없음 — 확인 필요"
    if user_sbiz in policy_sbiz or policy_sbiz in user_sbiz:
        return True, f"특수 분류 조건 충족 ({user_sbiz})"
    return False, f"특수 분류 불일치 (필요: {policy_sbiz}, 현재: {user_sbiz})"


def _check_region(user_region: str | None, policy: dict) -> tuple[bool, str]:
    policy_region = (policy.get("plcyAplyRgnCd") or "전국").strip()
    if not policy_region or policy_region in ("전국", "제한없음"):
        return True, "지역 제한 없음"
    if not user_region:
        return True, "거주 지역 정보 없음 — 확인 필요"
    if user_region in policy_region or policy_region in user_region:
        return True, f"지역 조건 충족 ({user_region})"
    return False, f"지역 불일치 (필요: {policy_region}, 현재: {user_region})"


@tool
async def check_eligibility(user_profile: dict, policy_metadata: dict) -> str:
    """
    사용자 정보와 복지 정책 자격 조건을 비교해 신청 가능 여부를 판별합니다.
    사용자가 특정 정책에 대해 "신청 가능한가요?"라고 물을 때 호출하세요.
    반드시 extract_user_profile을 먼저 호출해 user_profile을 준비하세요.

    Args:
        user_profile: extract_user_profile의 profile 값
        policy_metadata: 대상 정책의 메타데이터 딕셔너리

    Returns:
        str: 판정 결과 ("가능" / "불가" / "확인필요") + 항목별 사유
    """
    policy_name = policy_metadata.get("plcyNm", "해당 정책")
    logger.info("check_eligibility 실행: policy=%s", policy_name)

    age = user_profile.get("age") or calc_age(user_profile.get("birth_date", ""))

    checks = [
        _check_age(age, policy_metadata),
        _check_job(user_profile.get("jobcd"), policy_metadata),
        _check_income(user_profile.get("earncndsecd"), policy_metadata),
        _check_region(user_profile.get("zipcd"), policy_metadata),
        _check_school(user_profile.get("schoolcd"), policy_metadata),
        _check_sbiz(user_profile.get("sbizcd"), policy_metadata),
    ]

    failed   = [(ok, reason) for ok, reason in checks if ok is False]
    uncertain = [(ok, reason) for ok, reason in checks if "확인 필요" in reason or "정보 없음" in reason]

    lines = [f"[ {policy_name} ] 자격 판정 결과"]
    lines.append("-" * 50)
    for _, reason in checks:
        icon = "✅" if "충족" in reason or "제한없음" in reason else ("❌" if any(reason == f for _, f in failed) else "⚠️")
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
