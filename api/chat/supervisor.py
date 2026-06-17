import logging
from typing import Annotated
from fastapi import Depends
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import InMemorySaver

from api.chat.state import ShareState
from api.chat.nodes.analysis_node import analysis_node
from api.chat.nodes.route_node_fun import route_node_fun
from api.chat.nodes.housing_node import housing_node
from api.chat.nodes.employment_node import employment_node
from api.chat.nodes.education_node import education_node
from api.chat.nodes.finance_node import finance_node

_memory = InMemorySaver()  # 게스트용 싱글톤


class YouthPolicySupervisor:
    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.YouthPolicySupervisor")
        self.build_workflow()

    def build_workflow(self):
        graph = StateGraph(ShareState)

        graph.add_node("analysis",   analysis_node)
        graph.add_node("housing",    housing_node)
        graph.add_node("employment", employment_node)
        graph.add_node("education",  education_node)
        graph.add_node("finance",    finance_node)

        graph.add_edge(START, "analysis")
        graph.add_conditional_edges(
            "analysis",
            route_node_fun,
            {
                "housing":    "housing",
                "employment": "employment",
                "education":  "education",
                "finance":    "finance",
            }
        )

        graph.add_edge("housing",    END)
        graph.add_edge("employment", END)
        graph.add_edge("education",  END)
        graph.add_edge("finance",    END)

        self.work_flow = graph.compile(checkpointer=_memory)

    async def run(
        self,
        inquiry: str,
        conversation_id: str = "default",
        user_id: str | None = None,
    ) -> str:
        initial_state = ShareState(
            messages=[],
            user_inquiry=inquiry,
            user_id=user_id,
            conversation_id=conversation_id,
            inquiry_analysis="",
            final_response="아직 답변을 생성하지 않음",
        )
        config = {"configurable": {"thread_id": conversation_id}}
        final_state = await self.work_flow.ainvoke(
            initial_state, config=config
        )
        return final_state["final_response"]

    async def stream(
        self,
        inquiry: str,
        conversation_id: str = "default",
        user_id: str | None = None,
    ):
        initial_state = ShareState(
            messages=[],
            user_inquiry=inquiry,
            user_id=user_id,
            conversation_id=conversation_id,
            inquiry_analysis="",
            final_response="아직 답변을 생성하지 않음",
        )
        config = {"configurable": {"thread_id": conversation_id}}
        async for chunk in self.work_flow.astream(
            initial_state, config=config
        ):
            for node_output in chunk.values():
                if node_output.get("final_response"):
                    yield node_output["final_response"]

    def get_graph_image(self):
        return self.work_flow.get_graph().draw_mermaid_png()


YouthPolicySupervisorDep = Annotated[YouthPolicySupervisor, Depends(YouthPolicySupervisor)]
