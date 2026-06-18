import logging
from ..state import ShareState
from ..agents.welfare_agent import WelfareAgent

logger = logging.getLogger(__name__)
agent = WelfareAgent()

async def welfare_node(state: ShareState) -> dict:
    logger.info("복지문화 정책 노드 실행")
    result = await agent.run(state["user_inquiry"])
    return {"welfare_result": result}