from typing import Any
from langchain_core.tools import tool

@tool(return_direct=True)
async def compare_policies(policy_list: list[dict]) -> str:
    """
    여러 정책을 한눈에 비교할 수 있는 표 형태로 변환합니다.
    검색 결과가 2개 이상일 때 호출하세요.
    결과는 LLM 재가공 없이 바로 사용자에게 전달됩니다.

    Args:
        policy_list: 비교할 정책 딕셔너리 목록

    Returns:
        str: 정책명, 지원 대상, 혜택, 마감일, 신청 URL을 포함한 비교표 텍스트
    """

@tool
async def policy_priority_score(user_profile: dict, policy_list: list[dict]) -> str:
    """
    사용자 정보와 정책 조건을 비교해 각 정책의 적합도 점수를 계산하고 순위를 매깁니다.
    최종 답변을 생성하기 직전에 호출해 가장 적합한 정책을 상위에 노출하세요.

    Args:
        user_profile: 사용자 정보 딕셔너리
        policy_list: 점수를 매길 정책 목록

    Returns:
        str: 정책별 점수와 추천 순위 텍스트
    """