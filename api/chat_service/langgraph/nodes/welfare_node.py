import logging

from api.chat_service.langgraph.agents.welfare_agent import WelfareAgent
from api.chat_service.langgraph.state import ShareState


# 로거 생성
logger = logging.getLogger(__name__)

# 모듈 싱글톤 에이전트 생성 (파일이 import될 때 1회만 실행됨)
# - CLAUDE.md 11-1 패턴: 노드는 에이전트를 매 요청마다 새로 만들지 않고 재사용
agent = WelfareAgent()


async def welfare_node(state: ShareState) -> dict:
    """복지(금융·문화·예술) 분야 노드.

    슈퍼바이저가 category="복지"로 분류한 뒤 라우팅하면 이 노드가 실행된다.
    state의 user_query/user_role/user_profile을 풀어서 WelfareAgent.run()에
    그대로 전달하고, 에이전트의 응답을 final_response로 state에 반영한다.
    """
    logger.info("복지 노드 실행")

    response = await agent.run(
        user_query=state["user_query"],
        user_role=state.get("user_role", "guest"),
        user_profile=state.get("user_profile"),
    )

    logger.info(f"복지 노드 응답: {response[:100]}...")

    # 상태 업데이트: 최종 답변 저장
    return {"final_response": response}