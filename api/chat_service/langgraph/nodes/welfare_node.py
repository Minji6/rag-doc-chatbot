import logging
from ..state import ShareState, DomainResult
from ..constants import agent_mode
from ..agents.welfare_agent import WelfareAgent, _CATEGORY

logger = logging.getLogger(__name__)
agent = WelfareAgent()


async def welfare_node(state: ShareState) -> dict:
    """복지문화 정책 생성 노드.

    검색 노드(welfare_search_node)가 state.domain_knowledge/policies["welfare"]에 넣어둔 값을 읽어
    WelfareAgent에 전달하고, DomainResult를 받아 state.domain_results["welfare"]에 기록한다.
    """
    logger.info("복지문화 정책 생성 노드 실행")

    inquiry = state["user_inquiry"]
    image_context = state.get("image_context", "")
    if image_context:
        inquiry = f"{inquiry}\n\n[첨부 이미지 내용]\n{image_context}"

    text, policies, source, merged_profile, suggestions = await agent.run(
        inquiry=inquiry,
        knowledge=state.get("domain_knowledge", {}).get("welfare", ""),
        policies=state.get("domain_policies", {}).get("welfare", []),
        user_role=state.get("user_role", "guest"),
        user_profile=state.get("user_profile"),
        inquiry_type=agent_mode(state.get("inquiry_type", [])),
        messages=state.get("messages", []),
    )
    result = DomainResult(text=text, policies=policies, category=_CATEGORY, source=source)
    return {
        "domain_results": {"welfare": result},
        "user_profile": merged_profile,
        "suggestions": suggestions,
    }
