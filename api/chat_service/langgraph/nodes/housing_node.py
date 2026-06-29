import logging

from api.chat_service.langgraph.constants import AGENT_CATEGORY, agent_mode
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

    inquiry = state["user_inquiry"]
    image_context = state.get("image_context", "")
    if image_context:
        inquiry = f"{inquiry}\n\n[첨부 이미지 내용]\n{image_context}"

    text, suggestions = await agent.run(
        inquiry=inquiry,
        knowledge=knowledge,
        user_profile=state.get("user_profile"),
        inquiry_type=agent_mode(state.get("inquiry_type", [])),
    )

    source = "rag" if policies else "none"
    result = DomainResult (
        text = text, policies=policies, category=_CATEGORY, source=source
    )

    return {"domain_results": {"housing": result}, "suggestions": suggestions}
