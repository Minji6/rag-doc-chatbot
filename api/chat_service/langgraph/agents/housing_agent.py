import logging
from typing import Annotated
from fastapi import Depends
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_postgres import PGVector
from langchain.embeddings import init_embeddings
from api.common.sqlalchemy_conf import engine
from ..state import DomainResult, empty_domain_result
from ..constants import (
    AGENT_CATEGORY,
    PGVECTOR_COLLECTION_NAME,
    POLICY_METADATA_FIELDS,
    SIMILARITY_DISTANCE_THRESHOLD,
)

logger = logging.getLogger(__name__)

_CATEGORY = AGENT_CATEGORY["housing"]


def _pick_policy_fields(metadata: dict) -> dict:
    """PGVector 메타에서 화이트리스트 필드만 추려 dict 생성."""
    return {key: metadata.get(key) for key in POLICY_METADATA_FIELDS}


##############################################################
# 도구 정의
##############################################################
# 에이전트 실행 중 호출된 search_policy의 raw 결과를 노드 레벨에서 회수하기 위해
# 모듈 전역에 마지막 호출 결과를 캐싱한다. (LangChain agent는 tool의 raw return을 외부로 직접 노출하지 않음)
# 동시 호출 안전성은 도메인 에이전트가 노드 1회 호출당 1회 search_policy 호출이라는 가정에 의존.
_last_search_policies: list[dict] = []


@tool
async def search_policy(query: str) -> str:
    """
    주거 분야 청년 정책을 PGVector에서 검색합니다.
    Args:
        query: 검색할 질문이나 키워드
    Returns:
        str: 검색된 정책 내용
    """
    global _last_search_policies
    _last_search_policies = []

    vectorstore = PGVector(
        embeddings=init_embeddings(model="openai:text-embedding-3-large"),
        collection_name=PGVECTOR_COLLECTION_NAME,
        connection=engine,
        async_mode=True,
    )
    results = await vectorstore.asimilarity_search_with_score(
        query, k=5, filter={"category": _CATEGORY}
    )
    documents = [(doc, dist) for doc, dist in results if dist <= SIMILARITY_DISTANCE_THRESHOLD]

    if not documents:
        return f"'{query}' 관련 {_CATEGORY} 정책을 찾을 수 없습니다."

    _last_search_policies = [_pick_policy_fields(doc.metadata) for doc, _ in documents]

    lines = []
    for idx, (doc, _dist) in enumerate(documents, 1):
        lines.append(f"[정책 {idx}] {doc.metadata.get('plcyNm', '')}")
        lines.append(f"내용: {doc.page_content}")
        lines.append(f"신청 URL: {doc.metadata.get('aplyUrlAddr', '정보 없음')}\n")
    return "\n".join(lines)


##############################################################
# Agent 클래스 정의
##############################################################
class HousingAgent:
    def __init__(self, model: str = "openai:gpt-4o-mini") -> None:
        self.logger = logging.getLogger(f"{__name__}.HousingAgent")
        self.agent = create_agent(
            model=model,
            tools=[search_policy],
            system_prompt="당신은 청년 주거 정책 전문가입니다. 주거 관련 정책만 안내하세요.",
        )

    async def run(self, question: str) -> DomainResult:
        global _last_search_policies
        _last_search_policies = []   # 호출 시작 시 초기화

        result = await self.agent.ainvoke(
            {"messages": [{"role": "user", "content": question}]}
        )
        text = result["messages"][-1].content
        policies = list(_last_search_policies)   # 스냅샷
        source = "rag" if policies else "none"
        return DomainResult(text=text, policies=policies, category=_CATEGORY, source=source)


HousingAgentDep = Annotated[HousingAgent, Depends(HousingAgent)]
