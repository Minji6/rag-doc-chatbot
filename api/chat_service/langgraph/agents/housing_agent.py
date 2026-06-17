import logging
from typing import Annotated
from fastapi import Depends
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_postgres import PGVector
from langchain.embeddings import init_embeddings
from api.common.sqlalchemy_conf import engine

# 로거 생성
logger = logging.getLogger(__name__)

##############################################################
# 도구 정의
##############################################################
@tool
async def search_policy(query: str) -> str :
    """
    주거 분야 청년 정책을 PGVector에서 검색합니다.
    Args:
        query: 검색할 질문이나 키워드
    Returns:
        str: 검색된 정책 내용
    """
    vectorstore = PGVector(
        # 텍스트 -> 숫자 변환을 위한 모델 임베딩
        embeddings=init_embeddings(model="openai:text-embedding-3-large"),
        collection_name="울산_청년정책", #추후 컬렉션 내용 변경
        connection=engine,
        async_mode=True
    )
    
    # 카테고리가 '주거'인 항목중에서 유사도 기반 검색
    results = await vectorstore.asimilarity_search_with_score(
        query,
        k=5,
        filter={"category": "주거"}
    )
    
    # 임시 임계값 (데이터 교체 후 변경 가능)
    documents = [(doc, dist) for doc, dist in results if dist < 0.65]
    
    if not documents:
        return f"'{query}' 관련 주거 정책을 찾을 수 없습니다."
    
    lines = []
    for idx, (doc, dist) in enumerate(documents, 1):
        lines.append(f"[정책: {idx}] {doc.metadata.get('title', '')}")
        lines.append(f"내용: {doc.page_content}\n")
        
    return "\n".join(lines)
        
##############################################################
# Agent 클래스 정의
##############################################################

class HousingAgent:
    def __init__(self, model:str="openai:gpt-4o-mini") -> None:
        self.logger = logging.getLogger(f"{__name__}.HousingAgent")
        self.agent = create_agent(
            model = model,
            tools=[search_policy],
            system_prompt="당신은 청년 주거 정책 전문가입니다. 주거 관련 정책만 안내하세요."
        )
        
    async def run(self, question: str) -> str:
        result = await self.agent.ainvoke(
            {"messages": [{"role": "user", "content": question}]}

        )
        
        return result["messages"][-1].content
    
# 의존성 주입을 위한 타입 힌트 정의
HousingAgentDep = Annotated[HousingAgent, Depends(HousingAgent)]