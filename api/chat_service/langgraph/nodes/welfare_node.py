import logging
from ..state import ShareState, DomainResult
from ..agents.welfare_agent import WelfareAgent, _CATEGORY

logger = logging.getLogger(__name__)
agent = WelfareAgent()


async def welfare_node(state: ShareState) -> dict:
    """복지문화 정책 생성 노드.

    검색 노드(welfare_search_node)가 state에 넣어둔 welfare_knowledge_base/welfare_policies를 읽어
    WelfareAgent에 전달하고, DomainResult를 받아 state에 기록한다.
    """
    logger.info("복지문화 정책 생성 노드 실행")

    text, policies, source, merged_profile = await agent.run(
        inquiry=state["user_inquiry"],
        knowledge=state.get("welfare_knowledge_base", ""),
        policies=state.get("welfare_policies", []),
        user_role=state.get("user_role", "guest"),
        user_profile=state.get("user_profile"),
        messages=state.get("messages", []),
    )
    result = DomainResult(text=text, policies=policies, category=_CATEGORY, source=source)
    return {"welfare_result": result, "user_profile": merged_profile}
