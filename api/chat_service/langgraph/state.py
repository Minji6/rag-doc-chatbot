from typing import Annotated, TypedDict
from langgraph.graph import add_messages


class ShareState(TypedDict):
    messages: Annotated[list, add_messages]  # 필수
    user_inquiry: str         # 사용자 원본 질문
    user_id: str              # 사용자 ID (게스트면 빈 문자열)
    is_authenticated: bool    # 로그인 여부
    inquiry_analysis: str     # 의도 분석 결과
    housing_result: str       # 주거 에이전트 답변
    employment_result: str    # 취업 에이전트 답변
    education_result: str     # 교육 에이전트 답변
    finance_result: str       # 금융·문화·예술 에이전트 답변
    final_response: str       # 최종 답변
