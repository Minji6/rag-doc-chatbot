from langchain.tools import tool
from ._stub_agent import StubAgent
from ..constants import AGENT_CATEGORY


class EmploymentAgent(StubAgent):
    def __init__(self, model: str = "openai:gpt-4o-mini") -> None:
        super().__init__(domain_label="일자리", category=AGENT_CATEGORY["employment"])


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
    pass  # TODO: 구현 예정
