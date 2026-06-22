import logging
from ..state import ShareState
from ..agents.education_agent import EducationAgent

logger = logging.getLogger(__name__)
agent = EducationAgent()

async def education_node(state: ShareState) -> dict:
    logger.info("교육 정책 노드 실행")
    result = await agent.run(state["user_inquiry"], state["user_profile"])
    return {"education_result": result}