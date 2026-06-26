import logging
from typing import Annotated
from fastapi import Depends
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from api.common.psycopg_pool_conf import psycopg_pool

# 로거 생성
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Agent 클래스 정의
# ---------------------------------------------------------
class HistoryPostgreSQLAgent:
    def __init__(self, model:str="openai:gpt-4o-mini") -> None:
        self.logger = logging.getLogger(f"{__name__}.HistoryPostgreSQLAgent")
        self.model = model
        self.checkpointer = None
        self.agent = None

        # 비동기 초기화 여부
        self._initialized = False

    # 비동기 초기화 메소드
    async def _initialize(self):
        if self._initialized:
            return

        # self.checkpointer 초기화
        self.checkpointer = AsyncPostgresSaver(psycopg_pool) # type: ignore

        # 대화 기억을 저장할 테이블 자동 생성(최초 1회)
        await self.checkpointer.setup()

        # Agent 객체 생성
        self.agent = create_agent(
            model= self.model,
            checkpointer= self.checkpointer
        )

        # 비동기 초기화 완료 설정
        self._initialized = True
        # 주: 이 에이전트의 LLM은 추론에 쓰이지 않는다. checkpointer(AsyncPostgresSaver)를
        #     thread_id 기준으로 읽고/쓰는 저장소 용도로만 사용한다.
        #     (대화 답변 생성은 supervisor가 담당 — 여기서 LLM 호출 없음)

    # 대화 히스토리 조회
    async def get_history(self, thread_id:str) -> dict:
        await self._initialize()

        # 현재 에이전트의 상태(저장정보) 가져오기
        state = await self.agent.aget_state({ # type: ignore
            "configurable": {"thread_id": thread_id}
        })

        # 에이전트 상태에서 지난 과거 대화 내용이 없는 경우
        if not state or not state.values.get("messages"):
            return {
                "conversation_id": thread_id,
                "messages": []
            }

        # 에이전트 상태에서 지난 과거 대화 내용이 있는 경우
        messages = []
        for msg in state.values["messages"]:
            messages.append({
                "role": msg.type,
                "content": msg.content
            })

        return {
            "conversation_id": thread_id,
            "messages": messages
        }

    # 대화 히스토리 삭제하기
    async def clear_history(self, thread_id:str) -> dict:
        # 비동기 초기화 메소드 호출
        await self._initialize()

        # 해당 대화 ID를 완전히 삭제
        await self.checkpointer.adelete_thread(thread_id) # type: ignore

        return{
            "conversation_id": thread_id,
            "message": "대화 기록이 삭제되었습니다."
        }
        
    # supervisor가 생성한 답변을 LLM 재호출 없이 checkpointer에 직접 저장
    async def save_exchange(self, user_msg:str, ai_response: str, thread_id: str):
        await self._initialize() # 비동기 초기화 함수
        await self.agent.aupdate_state( # type: ignore
            {"configurable": {"thread_id": thread_id}},
            {"messages": [
                HumanMessage(content=user_msg),
                AIMessage(content=ai_response)
            ]},
            as_node="model",
        )


# 의존성 주입을 위한 타입 힌트 정의
HistoryPostgreSQLAgentDep = Annotated[HistoryPostgreSQLAgent, Depends(HistoryPostgreSQLAgent)]
