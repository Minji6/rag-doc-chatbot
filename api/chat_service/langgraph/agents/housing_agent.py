import logging
from contextvars import ContextVar
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
# search_policy의 raw 정책 결과를 노드 레벨로 회수하기 위한 채널.
# LangChain create_agent는 tool의 raw return을 외부에 직접 노출하지 않으므로 우회 통로 필요.
# ContextVar는 asyncio task별로 자동 격리되어 동시 호출 시에도 결과가 섞이지 않음.
_last_search_policies: ContextVar[list[dict]] = ContextVar(
    "housing_last_search_policies", default=[]
)


@tool
async def search_policy(query: str) -> str:
    """
    주거 분야 청년 정책을 PGVector에서 검색합니다.
    Args:
        query: 검색할 질문이나 키워드
    Returns:
        str: 검색된 정책 내용
    """
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
        _last_search_policies.set([])
        return f"'{query}' 관련 {_CATEGORY} 정책을 찾을 수 없습니다."

    _last_search_policies.set([_pick_policy_fields(doc.metadata) for doc, _ in documents])

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
        token = _last_search_policies.set([])   # 이 task 컨텍스트만 초기화
        try:
            result = await self.agent.ainvoke(
                {"messages": [{"role": "user", "content": question}]}
            )
            policies = _last_search_policies.get()
        finally:
            _last_search_policies.reset(token)

        text = result["messages"][-1].content
        source = "rag" if policies else "none"
        return DomainResult(text=text, policies=policies, category=_CATEGORY, source=source)


HousingAgentDep = Annotated[HousingAgent, Depends(HousingAgent)]
