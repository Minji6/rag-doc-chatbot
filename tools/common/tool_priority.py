from typing import Any
from langchain_core.tools import tool

@tool
async def answer_with_priority(query: str, category: str) -> str:
    """
    RAG를 1순위로 검색하고, 결과가 없을 경우 LLM 지식으로 답변합니다.
    모든 에이전트의 기본 답변 생성에 사용하세요.

    Args:
        query: 사용자 질문
        category: 정책 분야 ("복지문화" / "일자리" / "주거" / "교육")

    Returns:
        str: RAG 검색 결과 또는 LLM 기반 답변
    """