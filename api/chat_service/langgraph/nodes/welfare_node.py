import logging
from ..state import ShareState
from ..agents.welfare_agent import WelfareAgent

logger = logging.getLogger(__name__)
agent = WelfareAgent()


async def welfare_node(state: ShareState) -> dict:
    logger.info(
        "복지문화 정책 노드 실행 — user_role=%s",
        state.get("user_role", "guest"),
    )
    result = await agent.run(
        question=state["user_inquiry"],
        user_role=state.get("user_role", "guest"),
        user_profile=state.get("user_profile") or {},
        inquiry_type=state.get("inquiry_type", ""),
        history=state.get("messages", []),
    )
    return {"welfare_result": result}
