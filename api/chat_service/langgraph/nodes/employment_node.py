import logging
from ..state import ShareState
from ..agents.employment_agent import EmploymentAgent

logger = logging.getLogger(__name__)
agent = EmploymentAgent()

async def employment_node(state: ShareState) -> dict:
    logger.info("일자리 정책 노드 실행")
    result = await agent.run(state["user_inquiry"])
    return {"employment_result": result}