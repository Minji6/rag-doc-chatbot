import logging
from ..state import ShareState, DomainResult
from ..agents.employment_agent import EmploymentAgent
from ..constants import AGENT_CATEGORY

logger = logging.getLogger(__name__)
agent = EmploymentAgent()

_CATEGORY = AGENT_CATEGORY["employment"]


async def employment_node(state: ShareState) -> dict:
    """취업 정책 생성 노드.

    검색 노드(employment_search_node)가 state.domain_knowledge/policies["employment"]에 넣어둔
    값을 읽어 답변을 생성하고 DomainResult로 반환한다. (검색은 하지 않음)
    """
    logger.info("취업 정책 생성 노드 실행")

    knowledge = state.get("domain_knowledge", {}).get("employment", "")
    policies = state.get("domain_policies", {}).get("employment", [])

    inquiry = state["user_inquiry"]
    image_context = state.get("image_context", "")
    if image_context:
        inquiry = f"{inquiry}\n\n[첨부 이미지 내용]\n{image_context}"

    text = await agent.run(
        inquiry=inquiry,
        knowledge=knowledge,
        user_profile=state.get("user_profile"),
        inquiry_type=state.get("inquiry_type", "검색"),
    )

    source = "rag" if policies else "none"
    result = DomainResult(
        text=text, policies=policies, category=_CATEGORY, source=source
    )
    return {"domain_results": {"employment": result}}
