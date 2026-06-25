import logging
from ..state import ShareState, DomainResult
from ..agents.welfare_agent import WelfareAgent, _CATEGORY

logger = logging.getLogger(__name__)
agent = WelfareAgent()


async def welfare_node(state: ShareState) -> dict:
    """복지문화 정책 생성 노드.

    검색 노드(welfare_search_node)가 state.domain_knowledge/policies["welfare"]에 넣어둔 값을 읽어
    WelfareAgent에 전달하고, DomainResult를 받아 state.domain_results["welfare"]에 기록한다.
    """
    logger.info("복지문화 정책 생성 노드 실행")

    text, policies, source, merged_profile = await agent.run(
        inquiry=state["user_inquiry"],
        knowledge=state.get("domain_knowledge", {}).get("welfare", ""),
        policies=state.get("domain_policies", {}).get("welfare", []),
        user_role=state.get("user_role", "guest"),
        user_profile=state.get("user_profile"),
    )
    result = DomainResult(text=text, policies=policies, category=_CATEGORY, source=source)
    return {"domain_results": {"welfare": result}, "user_profile": merged_profile}
