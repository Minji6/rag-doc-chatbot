# api/chat_service/langgraph/supervisor.py
import logging
from typing import Annotated, Hashable
from fastapi import Depends
from langgraph.graph import END, START, StateGraph

from .state import ShareState, empty_domain_result
from .constants import AGENT_CATEGORY, CATEGORY_ROUTING
from .nodes.analysis_node import analysis_node
from .nodes.housing_node import housing_node
from .nodes.employment_search_node import employment_search_node
from .nodes.employment_node import employment_node
from .nodes.education_search_node import education_search_node
from .nodes.education_node import education_node
from .nodes.welfare_search_node import welfare_search_node
from .nodes.welfare_node import welfare_node
from .nodes.route_node_fun import route_node_fun
from .nodes.gather_node import gather_node

logger = logging.getLogger(__name__)


class ChatbotSupervisor:
    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.ChatbotSupervisor")
        self._build()

    def _build(self) -> None:
        graph = StateGraph(ShareState)

        # 노드 등록
        graph.add_node("analysis", analysis_node)
        graph.add_node("housing", housing_node)
        graph.add_node("employment_search", employment_search_node)
        graph.add_node("employment", employment_node)
        graph.add_node("education_search", education_search_node)  # 교육: 검색 노드
        graph.add_node("education", education_node)                # 교육: 생성 노드
        graph.add_node("welfare_search", welfare_search_node)      # 복지: 검색 노드
        graph.add_node("welfare", welfare_node)                    # 복지: 생성 노드
        graph.add_node("gather", gather_node)

        # 시작 → 의도 분석
        graph.add_edge(START, "analysis")

        # 분석 결과(category)에 따라 도메인 노드로 분기.
        # 교육/취업은 검색/생성을 분리했으므로 각 검색 노드로 먼저 보낸다.
        # (education_search → education → gather)
        # (employment_search → employment → gather)
        node_names = set(CATEGORY_ROUTING.values())
        route_map: dict[Hashable, str] = {name: name for name in node_names}
        route_map["education"] = "education_search"
        route_map["employment"] = "employment_search"
        route_map["welfare"] = "welfare_search"
        graph.add_conditional_edges(
            "analysis",
            route_node_fun,
            route_map,
        )

        # 검색 노드 → 생성 노드
        graph.add_edge("education_search", "education")
        graph.add_edge("employment_search", "employment")
        graph.add_edge("welfare_search", "welfare")

        # 도메인 노드 → 전부 gather로 모임 (명세서 개요 구조)
        for node_name in node_names:
            graph.add_edge(node_name, "gather")

        graph.add_edge("gather", END)

        self.workflow = graph.compile()

    async def run(
        self, 
        user_inquiry: str, 
        user_role: str = "guest", 
        user_profile: dict | None = None,
        messages: list | None = None, # 이전 대화 맥락 (없으면 첫 대화)
    ) -> str:
        
        user_profile = user_profile or {}
        initial_state = ShareState(
            messages= messages or [],
            user_inquiry=user_inquiry,
            user_role=user_role,
            user_profile=user_profile,
            category="",
            inquiry_type="",
            knowledge_base="",
            policies=[],
            welfare_knowledge_base="",
            welfare_policies=[],
            housing_result=empty_domain_result(AGENT_CATEGORY["housing"]),
            employment_result=empty_domain_result(AGENT_CATEGORY["employment"]),
            education_result=empty_domain_result(AGENT_CATEGORY["education"]),
            welfare_result=empty_domain_result(AGENT_CATEGORY["welfare"]),
            final_response="",
        )
        final_state = await self.workflow.ainvoke(initial_state)
        return final_state["final_response"]

    def get_graph_image(self) -> bytes:
        return self.workflow.get_graph().draw_mermaid_png()


ChatbotSupervisorDep = Annotated[ChatbotSupervisor, Depends(ChatbotSupervisor)]