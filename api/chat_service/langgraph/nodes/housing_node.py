import logging

from api.chat_service.langgraph.constants import AGENT_CATEGORY
from ..state import DomainResult, ShareState
from ..agents.housing_agent import HousingAgent

logger = logging.getLogger(__name__)
agent = HousingAgent()   # 모듈 싱글톤

_CATEGORY = AGENT_CATEGORY["housing"]

async def housing_node(state: ShareState) -> dict:
    """주거 정책 생성 노드. domain_knowledge["housing"] / domain_policies["housing"]을 읽어 답변 생성."""
    logger.info("주거 정책 생성 노드 실행")
    
    knowledge = state.get("domain_knowledge", {}).get("housing", "")
    policies = state.get("domain_policies", {}).get("housing", [])
    
    text = await agent.run(
        inquiry=state["user_inquiry"],
        knowledge=knowledge,
        user_profile=state.get("user_profile")
    )
    
    source = "rag" if policies else "none"
    result = DomainResult (
        text = text, policies=policies, category=_CATEGORY, source=source
    )
    
    return {"domain_results": {"housing": result}}