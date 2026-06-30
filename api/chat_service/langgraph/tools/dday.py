"""D-day(신청 마감) 공통 계산 — 이전엔 welfare_agent·employment_search·_policy_table에
세 벌로 중복돼 있던 로직을 한곳에 모은 모듈.

- parse_dates / end_date_from: 다양한 날짜 표기(YYYY.MM.DD, YYYYMMDD, 기간 범위)에서 마감일 추출
- dday_label: 사람이 읽는 D-day 라벨 (정책 표/검색 결과 공통)
- calculate_dday: LLM이 호출하는 @tool (welfare 등 에이전트가 import)
"""
import logging
import re
from datetime import date

from langchain.tools import tool

logger = logging.getLogger(__name__)

# YYYY.MM.DD / YYYY-MM-DD / YYYYMMDD 모두 매칭
_DATE_RE = re.compile(r"(\d{4})[.\-]?(\d{2})[.\-]?(\d{2})")


def parse_dates(text: str) -> list[date]:
    """문자열에서 파싱 가능한 모든 날짜를 추출한다 (실패 토큰은 무시)."""
    out: list[date] = []
    for m in _DATE_RE.finditer(text or ""):
        try:
            out.append(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        except ValueError:
            pass
    return out


def end_date_from(apply_ymd: str = "", biz_end_ymd: str = "") -> date | None:
    """신청기간/사업종료일 문자열에서 마감일(가장 늦은 날짜)을 구한다. 없으면 None."""
    dates = parse_dates(f"{apply_ymd or ''} {biz_end_ymd or ''}")
    return max(dates) if dates else None


def dday_label(
    apply_period_type: str = "",
    apply_ymd: str = "",
    biz_end_ymd: str = "",
    *,
    with_date: bool = True,
) -> str:
    """신청기간 구분 + 날짜로 D-day 라벨을 만든다.

    상시 → "상시접수", 마감 → "마감", 특정기간 → 날짜 파싱 후 D-N.
    with_date=True면 마감일(YYYY.MM.DD)을 괄호로 덧붙인다.
    날짜를 못 찾으면 "정보 없음".
    """
    se = (apply_period_type or "").strip()
    if se == "상시":
        return "상시접수"
    if se == "마감":
        return "마감"

    end = end_date_from(apply_ymd, biz_end_ymd)
    if end is None:
        return "정보 없음"

    delta = (end - date.today()).days
    stamp = end.strftime("%Y.%m.%d")
    if delta < 0:
        return f"마감 ({stamp})" if with_date else "마감"
    if delta == 0:
        return f"D-day ({stamp})" if with_date else "오늘 마감"
    return f"D-{delta} ({stamp})" if with_date else f"D-{delta}"


@tool
def calculate_dday(deadline: str, apply_period_type: str) -> str:
    """
    정책 신청 마감일까지 남은 일수를 계산합니다.
    사용자가 마감일을 물을 때 호출하세요.

    Args:
        deadline: 신청 마감일 (YYYYMMDD 형식, 없으면 빈 문자열)
        apply_period_type: 신청기간 구분 ("특정기간" / "상시" / "마감")
    Returns:
        str: D-day 문자열 (예: "D-30", "상시접수", "마감", "오늘 마감")
    """
    logger.info("calculate_dday 실행 — deadline=%s, apply_period_type=%s", deadline, apply_period_type)
    if (apply_period_type or "").strip() == "상시":
        return "상시접수"
    if not deadline:
        return "마감"
    # 날짜를 못 읽으면 "마감"으로 보수적 처리 (기존 welfare 동작 유지)
    if end_date_from(biz_end_ymd=deadline) is None:
        return "마감"
    return dday_label(apply_period_type, biz_end_ymd=deadline, with_date=False)
