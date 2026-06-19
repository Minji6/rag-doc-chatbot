from typing import Any
from langchain_core.tools import tool

@tool
async def search_policy(query: str, category: str, k: int = 5) -> str:
    """
    PGVector에서 청년 정책을 검색합니다. (RAG 1순위)
    사용자 질문이 들어오면 반드시 가장 먼저 이 도구를 호출하세요.
    결과가 없으면 [RAG_FALLBACK] 신호를 반환합니다.

    Args:
        query: 검색할 질문이나 키워드
        category: 정책 분야 ("복지문화" / "일자리" / "주거" / "교육")
        k: 반환할 최대 정책 수 (기본값 5)

    Returns:
        str: 검색된 정책 목록 텍스트. 결과 없으면 "[RAG_FALLBACK]" 신호 문자열
    """

@tool
async def search_web_supplement(
    query: str,
    exclude_titles: list[str],
    count: int,) -> str:
    """
    RAG 검색 결과가 부족할 때 웹 검색으로 정책을 보완합니다.
    반드시 search_policy를 먼저 호출한 뒤, 결과가 부족할 때만 사용하세요.
    exclude_titles에 포함된 정책과 다른 유사 정책만 반환합니다.

    Args:
        query: 검색할 질문이나 키워드
        exclude_titles: 이미 검색된 정책명 목록 (중복 제외용)
        count: 추가로 가져올 정책 수 (k - len(rag_results))

    Returns:
        str: 추가 정책 목록 텍스트
    """