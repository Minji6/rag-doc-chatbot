import logging
from ..state import ShareState, DomainResult
from ..agents.welfare_agent import WelfareAgent
from ..constants import AGENT_CATEGORY

logger = logging.getLogger(__name__)
agent = WelfareAgent()

_CATEGORY = AGENT_CATEGORY["welfare"]


async def welfare_node(state: ShareState) -> dict:
    """복지문화 정책 생성 노드.

    검색 노드(welfare_search_node)가 state에 넣어둔 welfare_knowledge_base/welfare_policies를 읽어
    답변을 생성하고 DomainResult로 반환한다. (검색은 하지 않음)
    """
    logger.info("복지문화 정책 생성 노드 실행")

    knowledge = state.get("welfare_knowledge_base", "")
    policies = state.get("welfare_policies", [])

    text = await agent.run(
        inquiry=state["user_inquiry"],
        knowledge=knowledge,
        user_role=state.get("user_role", "guest"),
        user_profile=state.get("user_profile"),
    )

    source = "rag" if policies else "none"
    result = DomainResult(
        text=text, policies=policies, category=_CATEGORY, source=source
    )
    return {"welfare_result": result}
