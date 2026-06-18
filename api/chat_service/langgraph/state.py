from typing import Annotated, TypedDict
from langgraph.graph import add_messages


class ShareState(TypedDict):
    messages: Annotated[list, add_messages]   # 필수 — 대화 누적

    user_inquiry: str      # 사용자 원본 질문 (지침서 §18-1 표준)
    user_role: str         # "guest" 또는 "user" (로그인 여부)

    category: str          # 분야: 주거 / 취업 / 교육 / 복지 (constants.py 참고)
    inquiry_type: str      # 의도: 검색 / 추천 / 상세조회 / 비교 (constants.py 참고)

    housing_result: str        # 주거 에이전트 답변
    employment_result: str     # 취업 에이전트 답변
    education_result: str      # 교육 에이전트 답변
    welfare_result: str        # 복지 에이전트 답변

    final_response: str        # gather_node가 채우는 최종 답변