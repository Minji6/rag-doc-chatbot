"""도메인 에이전트·노드가 공유하는 공통 툴 모음.

지금까지 각 에이전트/노드에 흩어져 있던 cross-cutting 로직(특히 D-day 계산)을 한곳에 모은다.
새 공통 툴은 여기에 추가하고 각 에이전트에서 import 한다.
"""
from .dday import calculate_dday, dday_label, end_date_from, parse_dates

__all__ = ["calculate_dday", "dday_label", "end_date_from", "parse_dates"]
