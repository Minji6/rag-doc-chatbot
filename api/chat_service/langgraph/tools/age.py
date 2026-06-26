"""만나이 계산 공통 유틸 — housing_search_node와 welfare_agent에 2벌로 중복돼 있던 로직 통합.

두 구현(문자열 슬라이스 방식 / date.fromisoformat 방식)을 동작 동등한 한 함수로 합친다.
'YYYY-MM-DD' 또는 그 뒤에 시간이 붙은 ISO 문자열, None/빈값/형식오류 모두 안전 처리.
"""
from datetime import date


def calc_age(birth_date: str | None) -> int | None:
    """생년월일 문자열로 만나이를 계산한다. 값이 없거나 형식이 잘못되면 None."""
    if not birth_date:
        return None
    try:
        y, m, d = map(int, str(birth_date)[:10].split("-"))
        born = date(y, m, d)
    except (ValueError, AttributeError, TypeError):
        return None
    today = date.today()
    # (월, 일) 비교로 생일이 안 지났으면 1살 빼기 (bool → 0/1)
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
