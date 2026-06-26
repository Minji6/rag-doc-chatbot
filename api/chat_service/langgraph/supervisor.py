# api/chat_service/langgraph/supervisor.py
import logging
from typing import Annotated, Hashable
from fastapi import Depends
from langgraph.graph import END, START, StateGraph

from api.chat_service.langgraph.nodes.housing_search_node import housing_search_node

from .state import ShareState
from .nodes.analysis_node import analysis_node
from .nodes.housing_node import housing_node
from .nodes.employment_search_node import employment_search_node
from .nodes.employment_node import employment_node
from .nodes.education_search_node import education_search_node
from .nodes.education_node import education_node
from .nodes.welfare_search_node import welfare_search_node
from .nodes.welfare_node import welfare_node
from .nodes.route_node_fun import route_node_fun
from .nodes.composer_node import composer_node
from .nodes.image_analysis_node import image_analysis_node
from .nodes.contextualize_node import contextualize_node

logger = logging.getLogger(__name__)


class ChatbotSupervisor:
    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.ChatbotSupervisor")
        self._build()

    def _build(self) -> None:
        graph = StateGraph(ShareState)

        # 노드 등록
        graph.add_node("image_analysis", image_analysis_node)
        graph.add_node("contextualize", contextualize_node)  # 후속질문 → 독립형 질의 재작성 + 참조 해소
        graph.add_node("analysis", analysis_node)
        graph.add_node("employment_search", employment_search_node)
        graph.add_node("employment", employment_node)
        graph.add_node("education_search", education_search_node)  # 교육: 검색 노드
        graph.add_node("education", education_node)                # 교육: 생성 노드
        graph.add_node("welfare_search", welfare_search_node)      # 복지: 검색 노드
        graph.add_node("welfare", welfare_node)                    # 복지: 생성 노드
        graph.add_node("composer", composer_node)                  # 조립: fragment → 최종 답변
        graph.add_node("housing_search", housing_search_node)  # 주거: 검색 노드
        graph.add_node("housing", housing_node)                # 주거: 생성 노드
        # 시작 → 이미지 분석 → 맥락화 → 의도 분석
        # (이미지 없으면 image_analysis 즉시 통과 / 첫 턴이면 contextualize도 LLM 없이 통과)
        graph.add_edge(START, "image_analysis")
        graph.add_edge("image_analysis", "contextualize")
        graph.add_edge("contextualize", "analysis")

        # 분석 결과(category list)에 따라 1개 또는 N개의 검색 노드로 fan-out.
        # route_node_fun이 list[str]을 반환하면 LangGraph가 자동으로 병렬 실행한다.
        # (예: ["주거","일자리"] → housing_search + employment_search 동시 실행)
        route_map: dict[Hashable, str] = {
            "employment": "employment_search",
            "housing":    "housing_search",
            "education":  "education_search",
            "welfare":    "welfare_search",
        }
        graph.add_conditional_edges(
            "analysis",
            route_node_fun,
            route_map, # type:ignore
        )

        # 검색 노드 → 생성 노드
        graph.add_edge("education_search", "education")
        graph.add_edge("employment_search", "employment")
        graph.add_edge("housing_search", "housing")  # 
        graph.add_edge("welfare_search", "welfare")

        # 도메인 생성 노드 → 전부 composer로 모임(fan-in). fan-out 시 활성화된 가지만 composer에 도달.
        for domain_node in ("employment", "housing", "education", "welfare"):
            graph.add_edge(domain_node, "composer")

        graph.add_edge("composer", END)

        self.workflow = graph.compile()

    async def run(
        self,
        user_inquiry: str,
        user_role: str = "guest",
        user_profile: dict | None = None,
        messages: list | None = None, # 이전 대화 맥락 (없으면 첫 대화)
        image_base64: str | None = None,
        image_content_type: str | None = None,
        last_policies: list[dict] | None = None,  # 직전 턴 정책 메타 (후속질문 해소용)
    ) -> dict:
        """챗봇 워크플로우 실행 후 프론트엔드 UI 분기에 필요한 메타까지 함께 반환.

        Returns:
            dict: {
                "message": str — composer가 조립한 최종 답변 텍스트,
                "category": list[str] — analysis_node가 분류한 분야 리스트 (멀티 가능),
                "inquiry_type": str — 의도 ("검색"/"추천"/"상세조회"/"비교"),
                "policies": list[dict] — 활성 도메인의 raw 정책 메타 (D-day 계산 등 프론트 처리용),
                "suggestions": list[str] — 교육 에이전트가 생성한 follow-up 질문,
            }
        """
        user_profile = user_profile or {}
        initial_state = ShareState(
            messages=messages or [],
            user_inquiry=user_inquiry,
            user_role=user_role,
            user_profile=user_profile,
            category=[],
            inquiry_type="",
            domain_knowledge={},
            domain_policies={},
            domain_results={},
            image_base64=image_base64,
            image_content_type=image_content_type,
            image_context="",
            last_policies=last_policies or [],
            resolved_policies=[],
            suggestions=[],
            final_response="",
        )
        final_state = await self.workflow.ainvoke(initial_state)

        # 응답 policies는 "답변이 다루는 정책"을 반영한다.
        # - 후속질문이 직전 정책을 특정했으면(resolved_policies) 그 정책만 노출
        #   → "두번째 정책 알려줘"에 5개가 딸려오던 문제 해결.
        # - 그 외에는 활성 도메인의 검색 결과를 합쳐서 노출 (검색·추천·비교).
        resolved = final_state.get("resolved_policies") or []
        if resolved:
            policies = resolved
        else:
            policies = []
            for result in final_state.get("domain_results", {}).values():
                policies.extend(result.get("policies", []))

        return {
            "message": final_state["final_response"],
            "category": final_state.get("category", []),
            "inquiry_type": final_state.get("inquiry_type", ""),
            "policies": policies,
            "suggestions": final_state.get("suggestions", []),
        }

    def get_graph_image(self) -> bytes:
        return self.workflow.get_graph().draw_mermaid_png()


ChatbotSupervisorDep = Annotated[ChatbotSupervisor, Depends(ChatbotSupervisor)]