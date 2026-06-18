import logging
from ..state import ShareState
from ..agents.housing_agent import HousingAgent

logger = logging.getLogger(__name__)
agent = HousingAgent()   # 모듈 싱글톤

async def housing_node(state: ShareState) -> dict:
    logger.info("주거 정책 노드 실행")
    result = await agent.run(state["user_inquiry"])
    return {"housing_result": result}