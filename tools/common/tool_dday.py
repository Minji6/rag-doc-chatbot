from datetime import date
from langchain_core.tools import tool

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
