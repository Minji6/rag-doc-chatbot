import logging
from ..state import ShareState, DomainResult
from ..agents.education_agent import EducationAgent
from ..constants import AGENT_CATEGORY

logger = logging.getLogger(__name__)
agent = EducationAgent()

_CATEGORY = AGENT_CATEGORY["education"]


async def education_node(state: ShareState) -> dict:
    """교육 정책 생성 노드.

    검색 노드(education_search_node)가 state에 넣어둔 knowledge_base/policies를 읽어
    답변을 생성하고 DomainResult로 반환한다. (검색은 하지 않음)
    """
    logger.info("교육 정책 생성 노드 실행")

    knowledge = state.get("knowledge_base", "")
    policies = state.get("policies", [])

    text = await agent.run(
        inquiry=state["user_inquiry"],
        knowledge=knowledge,
        policies=policies,
        user_profile=state.get("user_profile"),
        inquiry_type=state.get("inquiry_type", "검색"),
    )

    source = "rag" if policies else "none"
    result = DomainResult(
        text=text, policies=policies, category=_CATEGORY, source=source
    )
    return {"education_result": result}
