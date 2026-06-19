import logging
from datetime import date
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

@tool
def calculate_dday(deadline: str, apply_period_type: str) -> str:
    """
    정책 신청 마감일까지 남은 일수를 계산합니다.
    검색 결과에 마감일 정보가 있을 때 자동으로 호출하세요.
    오늘 날짜는 내부에서 자동으로 취득합니다.

    Args:
        deadline: 신청 마감일 (YYYYMMDD 형식, 상시 접수이면 빈 문자열 가능)
        apply_period_type: 신청기간 구분 ("특정기간" / "상시" / "마감")

    Returns:
        str: D-day 문자열 ("D-30", "상시접수", "오늘 마감", "마감" 등)
    """
    logger.info(f"calculate_dday 실행: deadline={deadline}, apply_period_type={apply_period_type}")

    if apply_period_type == "상시":
        return "상시접수"

    if not deadline:
        return "마감"

    today = date.today()
    try:
        deadline_date = date(
            int(deadline[:4]),
            int(deadline[4:6]),
            int(deadline[6:8])
        )
    except (ValueError, IndexError):
        logger.warning(f"마감일 파싱 실패: {deadline}")
        return "마감"

    diff = (deadline_date - today).days

    if diff < 0:
        return "마감"
    elif diff == 0:
        return "오늘 마감"
    else:
        return f"D-{diff}"
