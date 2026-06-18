import logging
from typing import Annotated, Literal

from fastapi import Depends, Form, HTTPException, Query

from api.history_service.agent_history_inmemory import HistoryInMemoryAgent
from api.history_service.agent_history_postgresql import HistoryPostgreSQLAgent


logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# 각 Agent는 모듈 싱글톤으로 관리
# ---------------------------------------------------------
_inmemory_agent = HistoryInMemoryAgent()
_postgresql_agent = HistoryPostgreSQLAgent()

def _select_agent(role: Literal["user", "guest"]) -> HistoryInMemoryAgent | HistoryPostgreSQLAgent:
    if role == "user":
        logger.info("role= user 이므로 HistoryPostgreSQLAgent 선택")
        return _postgresql_agent

    logger.info("role= guest 이므로 HistoryInMemoryAgent 선택")
    return _inmemory_agent

def get_history_agent_by_query(
    role: Annotated[Literal["user", "guest"], Query(description="user 또는 guest")]
) -> HistoryInMemoryAgent | HistoryPostgreSQLAgent:
    """
    GET/DELETE 요청용: role을 Query 파라미터로 받아 알맞은 에이전트를 반환합니다.
    """
    return _select_agent(role)

def get_history_agent_by_form(
    role: Annotated[Literal["user", "guest"], Form(description="user 또는 guest")]    
) -> HistoryInMemoryAgent | HistoryPostgreSQLAgent:
    """
    POST 요청용: role을 Form 파라미터로 받아 알맞은 에이전트를 반환합니다.
    """
    return _select_agent(role)

def build_thread_id(
    role: Literal["user", "guest"],
    conversation_id: str,
    user_id: str | None = None,
) -> str:
    """
    role에 따라 thread_id를 구성하고 검증합니다.
    user는 user_id를 합성해 회원 간 conversation_id 충돌을 방지하고,
    guest는 conversation_id 그대로 사용합니다(저장이 휘발성이라 충돌 리스크가 낮음).

    Raises:
        HTTPException: role이 user인데 user_id가 없는 경우 (422)
    """
    if role == "user":
        if not user_id:
            raise HTTPException(status_code=422, detail="user role에는 user_id가 필요합니다.")
        return f"{user_id}:{conversation_id}"
    return conversation_id

# ---------------------------------------------------------
# 의존성 타입 별칭
# ---------------------------------------------------------
HistoryAgentByQueryDep = Annotated[
    HistoryInMemoryAgent | HistoryPostgreSQLAgent, Depends(get_history_agent_by_query)
]
HistoryAgentByFormDep = Annotated[
    HistoryInMemoryAgent | HistoryPostgreSQLAgent, Depends(get_history_agent_by_form)
]