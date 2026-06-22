from typing import Any
from langchain.tools import tool

@tool
async def check_eligibility(user_profile: dict, policy_metadata: dict) -> str:
    """
    사용자 정보와 복지 정책 자격 조건을 비교해 신청 가능 여부를 판별합니다.
    사용자가 특정 정책에 대해 "신청 가능한가요?"라고 물을 때 호출하세요.
    반드시 extract_user_profile을 먼저 호출해 user_profile을 준비하세요.

    Args:
        user_profile: extract_user_profile로 추출한 사용자 정보 (profile 키 값)
        policy_metadata: 대상 정책의 메타데이터 딕셔너리

    Returns:
        str: 판정 결과 ("가능" / "불가" / "확인필요") + 사유 설명
    """
