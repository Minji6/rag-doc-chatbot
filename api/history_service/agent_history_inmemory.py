import logging
from typing import Annotated, cast
from fastapi import Depends
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from api.common.utils import LoggingCallbackHandler

# 로거 생성
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# InMemorySaver 생성, 모듈 싱글톤
# ---------------------------------------------------------
in_memory_saver = InMemorySaver()

# ---------------------------------------------------------
# Agent 클래스 정의
# ---------------------------------------------------------
class HistoryInMemoryAgent:
    def __init__(self, model:str="openai:gpt-4o-mini") -> None:
        self.logger = logging.getLogger(f"{__name__}.HistoryInMemoryAgent")
        self.agent = create_agent(
            model,
            checkpointer=in_memory_saver
        )
    
    # 에이전트 실행 메소드    
    async def run(self, message:str, thread_id: str) -> str:
        result = await self.agent.ainvoke(
            {"messages":[{"role":"user", "content":message}]},
            {"configurable": {"thread_id": thread_id},
             "callbacks": [LoggingCallbackHandler()]}
        )
        
        return result["messages"][-1].content
    
    # 대화 히스토리 조회
    async def get_history(self, thread_id:str) -> dict:
        # 현재 에이전트 상태 가져오기
        state = await self.agent.aget_state({
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
                "role":msg.type,
                "content": msg.content
            })
            
        return {
            "conversation_id": thread_id,
            "messages": messages
        }
        
    # 대화 히스토리 삭제
    async def clear_history(self, thread_id:str) -> dict:
        # 해당 대화 ID를 완전히 삭제
        checkpointer = cast(InMemorySaver, self.agent.checkpointer)
        await checkpointer.adelete_thread(thread_id)
        return {
            "conversation_id": thread_id,
            "message": "대화 기록이 삭제되었습니다."
        }
    
    # supervisor가 생성한 답변을 LLM 재호출 없이 checkpointer에 직접 저장
    async def save_exchange(self, user_msg: str, ai_response: str, thread_id: str):
        await self.agent.aupdate_state(
            {"configurable": {"thread_id": thread_id}},
            {"messages": [
                HumanMessage(content=user_msg),
                AIMessage(content=ai_response)
            ]},
            as_node="model",
        )
    
# 의존성 주입을 위한 타입 힌트 정의
HistoryInMemoryAgentDep = Annotated[HistoryInMemoryAgent, Depends(HistoryInMemoryAgent)]