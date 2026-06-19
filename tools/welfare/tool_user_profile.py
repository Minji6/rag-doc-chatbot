from typing import Any
from langchain_core.tools import tool

@tool
async def extract_user_profile(user_message: str) -> dict:
    """
    사용자 메시지에서 나이, 지역, 소득, 취업상태, 혼인 여부 등 정보를 추출합니다.
    자격 판별(check_eligibility)이나 맞춤 순위 산출(policy_priority_score) 전에 반드시 먼저 호출하세요.
    정보가 부족하면 follow_up_question으로 추가 질문을 생성합니다.
    missing_fields가 있으면 해당 질문을 먼저 사용자에게 전달하고, 나머지 도구 호출을 보류하세요.

    Args:
        user_message: 사용자 원본 메시지

    Returns:
        dict: profile(추출된 정보), missing_fields(부족한 필드), follow_up_question(추가 질문)
    """