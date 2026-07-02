"""복지 특화 자격 검증 툴 — welfare_node 전용.

상세조회 + 로그인 시, 복지 정책의 6조건(연령/지역/혼인/취업/소득/학력)을
구조화(dict)로 판정해 policy["eligibility"]에 첨부한다.
프론트 상세 모달의 '자격 검증' 섹션이 이 결과({status, items, summary})를 렌더한다.

검색 노드가 쓰는 eligibility.py의 3조건 튜플 검사(check_policy_eligibility)와 달리,
여기서는 취업/소득/학력을 포함한 6조건을 모달 렌더용 구조화 필드로 만든다.
게스트(프로필 없음)면 build_eligibility_report는 None을 반환한다.
"""
from typing import Literal

from .age import calc_age

# 조건별 상태 — 프론트 모달 렌더용(✅ 충족 met / ❌ 미충족 unmet / ⚠️ 확인 불가 unknown)
ConditionStatus = Literal["met", "unmet", "unknown"]


def _cond(label: str, requirement: str, user_value: str, status: ConditionStatus) -> dict:
    """상세 모달 렌더용 단일 조건 결과."""
    return {"label": label, "requirement": requirement, "userValue": user_value, "status": status}


def _age_req(lo: int | None, hi: int | None) -> str:
    if lo is not None and hi is not None:
        return f"만 {lo}~{hi}세"
    if lo is not None:
        return f"만 {lo}세 이상"
    if hi is not None:
        return f"만 {hi}세 이하"
    return "제한없음"


def _check_age(meta: dict, age: int | None) -> dict:
    user_val = f"만 {age}세" if age is not None else "정보 없음"
    if meta.get("sprtTrgtAgeLmtYn") == "N":
        return _cond("연령", "제한없음", user_val, "met")
    lo_raw, hi_raw = meta.get("sprtTrgtMinAge"), meta.get("sprtTrgtMaxAge")
    lo = int(lo_raw) if lo_raw else None
    hi = int(hi_raw) if hi_raw else None
    requirement = _age_req(lo, hi)
    if age is None:
        return _cond("연령", requirement, "정보 없음", "unknown")
    if lo is None and hi is None:
        return _cond("연령", "정보 없음", user_val, "unknown")
    if lo is not None and age < lo:
        return _cond("연령", requirement, user_val, "unmet")
    if hi is not None and age > hi:
        return _cond("연령", requirement, user_val, "unmet")
    return _cond("연령", requirement, user_val, "met")


def _check_region(meta: dict, user_region: str | None) -> dict:
    region_code = meta.get("zipCd") or ""
    if not region_code:
        return _cond("지역", "제한없음", user_region or "정보 없음", "met")
    if not user_region:
        return _cond("지역", region_code, "정보 없음", "unknown")
    # 양방향 substring — "전북" / "전라북도" / "전북특별자치도" 어떤 형태든 매칭
    parts = [p.strip() for p in region_code.split(",") if p.strip()]
    for p in parts:
        if p in user_region or user_region in p:
            return _cond("지역", region_code, user_region, "met")
    return _cond("지역", region_code, user_region, "unmet")


def _check_marriage(meta: dict, user_mrg: str | None) -> dict:
    pol_mrg = meta.get("mrgSttsCd")
    if pol_mrg == "제한없음":
        return _cond("혼인", "제한없음", user_mrg or "정보 없음", "met")
    if not pol_mrg:
        return _cond("혼인", "정보 없음", user_mrg or "정보 없음", "unknown")
    if not user_mrg:
        return _cond("혼인", pol_mrg, "정보 없음", "unknown")
    if user_mrg == pol_mrg:
        return _cond("혼인", pol_mrg, user_mrg, "met")
    return _cond("혼인", pol_mrg, user_mrg, "unmet")


def _check_job(meta: dict, user_job: str | None) -> dict:
    policy_job = (meta.get("jobCd") or "제한없음").strip()
    if not policy_job or policy_job == "제한없음":
        return _cond("취업", "제한없음", user_job or "정보 없음", "met")
    if not user_job:
        return _cond("취업", policy_job, "정보 없음", "unknown")
    if user_job == policy_job:
        return _cond("취업", policy_job, user_job, "met")
    return _cond("취업", policy_job, user_job, "unmet")


def _to_manwon_bound(value) -> int | None:
    """정책 소득 경계값(만원)을 정수로. 0·빈값·파싱실패는 '경계 없음'(None)으로 취급."""
    try:
        v = int(value)
    except (ValueError, TypeError):
        return None
    return v if v > 0 else None


def _income_range_req(lo: int | None, hi: int | None) -> str:
    """소득 요건(만원) 표시 문자열."""
    if lo is not None and hi is not None:
        return f"{lo:,}만원 ~ {hi:,}만원"
    if hi is not None:
        return f"{hi:,}만원 이하"
    if lo is not None:
        return f"{lo:,}만원 이상"
    return "제한없음"


def _check_income(meta: dict, user_income) -> dict:
    """소득 조건 검사.

    정책: earnCndSeCd(무관/연소득/기타) 중 '연소득'일 때만 earnMinAmt~earnMaxAmt(만원)
          구간을 검사한다. 무관/기타/빈값은 소득 제한 없음.
    사용자: earncndsecd는 '원' 단위이므로 만원으로 환산(÷10000)해 비교·표시한다.
    """
    # 사용자 소득(원) → 만원 환산
    try:
        user_manwon = int(user_income) // 10000 if user_income not in (None, "") else None
    except (ValueError, TypeError):
        user_manwon = None
    user_display = f"{user_manwon:,}만원" if user_manwon is not None else "정보 없음"

    cnd = (meta.get("earnCndSeCd") or "").strip()
    # 무관/기타/빈값 → 소득 제한 없음
    if cnd != "연소득":
        return _cond("소득", "제한없음", user_display, "met")

    # 연소득 → earnMinAmt ~ earnMaxAmt (만원) 구간 검사. 0/빈값은 해당 경계 없음.
    lo = _to_manwon_bound(meta.get("earnMinAmt"))
    hi = _to_manwon_bound(meta.get("earnMaxAmt"))
    requirement = _income_range_req(lo, hi)

    if user_manwon is None:
        return _cond("소득", requirement, "정보 없음", "unknown")

    ok = (lo is None or user_manwon >= lo) and (hi is None or user_manwon <= hi)
    return _cond("소득", requirement, user_display, "met" if ok else "unmet")


def _check_school(meta: dict, user_school: str | None) -> dict:
    policy_school = (meta.get("schoolcd") or "제한없음").strip()
    if not policy_school or policy_school == "제한없음":
        return _cond("학력", "제한없음", user_school or "정보 없음", "met")
    if not user_school:
        return _cond("학력", policy_school, "정보 없음", "unknown")
    if user_school == policy_school:
        return _cond("학력", policy_school, user_school, "met")
    return _cond("학력", policy_school, user_school, "unmet")


def build_eligibility_report(user_profile: dict, policy_metadata: dict) -> dict | None:
    """상세조회 모달용 구조화 자격 검증 결과를 만든다. 게스트(프로필 없음)면 None.

    Returns:
        dict | None: {
            "status": "eligible" | "partial" | "ineligible",  # 🟢/🟡/🔴
            "items": [{label, requirement, userValue, status}, ...],  # 6개 조건
            "summary": str,  # 종합 결과 1~2문장
        }
    """
    if not user_profile:
        return None

    age = calc_age(user_profile.get("birth_date"))
    items = [
        _check_age(policy_metadata, age),
        _check_region(policy_metadata, user_profile.get("zipcd")),
        _check_marriage(policy_metadata, user_profile.get("mrgsttscd")),
        _check_job(policy_metadata, user_profile.get("jobcd")),
        _check_income(policy_metadata, user_profile.get("earncndsecd")),
        _check_school(policy_metadata, user_profile.get("schoolcd")),
    ]

    unmet = [c for c in items if c["status"] == "unmet"]
    has_unknown = any(c["status"] == "unknown" for c in items)

    if unmet:
        # 자격 요건은 하나라도 미충족이면 부적격 — 나머지를 충족했어도 신청 대상 아님(🔴).
        overall = "ineligible"
        blocking = ", ".join(c["label"] for c in unmet)
        summary = f"현재 {blocking} 요건을 충족하지 않아 신청 대상이 아닙니다."
    elif has_unknown:
        # 미충족은 없지만 정보 부족으로 확정 불가한 조건이 있음(🟡).
        overall = "partial"
        summary = "일부 조건은 정보가 부족해 자격을 확정할 수 없습니다. 추가 정보 확인이 필요합니다."
    else:
        overall = "eligible"
        summary = "모든 자격 요건을 충족하여 신청 가능합니다."

    return {"status": overall, "items": items, "summary": summary}
