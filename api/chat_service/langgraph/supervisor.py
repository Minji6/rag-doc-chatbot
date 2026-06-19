import logging
from typing import Annotated
from fastapi import Depends
from langgraph.graph import END, START, StateGraph
from .state import ShareState
from .nodes.employment_node import employment_node

logger = logging.getLogger(__name__)


class ChatbotSupervisor:
    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.ChatbotSupervisor")
        self._build()

    def _build(self):
        graph = StateGraph(ShareState)

        graph.add_node("employment", employment_node)

        graph.add_edge(START, "employment")  # 바로 취업 노드로
        graph.add_edge("employment", END)

        self.workflow = graph.compile()

    async def run(self, inquiry: str, user_id: str = "") -> str:
        initial_state = ShareState(
            messages=[],
            user_inquiry=inquiry,
            user_id=user_id,
            is_authenticated=bool(user_id),
            inquiry_analysis="",
            housing_result="",
            employment_result="",
            education_result="",
            finance_result="",
            final_response="",
        )
        final_state = await self.workflow.ainvoke(initial_state)
        return final_state["employment_result"]  # 취업 결과만 반환


# ── Dep 타입 별칭 (파일 하단) ─────────────────────────────
ChatbotSupervisorDep = Annotated[ChatbotSupervisor, Depends(ChatbotSupervisor)]