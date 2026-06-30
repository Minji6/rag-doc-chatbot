# api/chat_service/langgraph/supervisor.py
import asyncio
import logging
from typing import Annotated, Hashable
from fastapi import Depends
from langchain.chat_models import init_chat_model
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
from .nodes.general_node import general_node
from .nodes.image_analysis_node import image_analysis_node
from .nodes.contextualize_node import contextualize_node
from api.chat_service.policy_memory import get_last_policies, save_last_policies

logger = logging.getLogger(__name__)

_enrich_llm = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0.0)

_ENRICH_SYSTEM_PROMPT = """당신은 청년정책 안내문 요약 AI입니다.
아래 정책의 '지원 내용' 원문을 2~3문장의 짧은 요약문으로 작성하세요.

규칙:
- 2~3문장 이내로 핵심(지원 금액·기간·지원 방식·주요 조건)만 남기세요.
- 세부 절차나 부가 조건은 생략해도 됩니다.
- 존댓말(~합니다/~됩니다)을 사용하세요.
- 서식(제목·불릿·번호·마크다운) 없이 평문으로만 쓰세요.
- "신청하세요", "활용해 보세요" 같은 권유·홍보 문구는 쓰지 마세요.
- 원문에 없는 정보를 추가하거나 지어내지 마세요."""


async def _summarize_support_content(plcy_sprt_cn: str, plcy_nm: str) -> str:
    """지원 내용 원문을 LLM으로 가독성 좋게 정리한다."""
    if not plcy_sprt_cn or not plcy_sprt_cn.strip():
        return ""
    try:
        result = await _enrich_llm.ainvoke([
            {"role": "system", "content": _ENRICH_SYSTEM_PROMPT},
            {"role": "user", "content": f"정책명: {plcy_nm}\n\n지원 내용 원문:\n{plcy_sprt_cn}"},
        ])
        return str(result.content).strip()
    except Exception:
        logger.exception("지원내용 LLM 정리 실패 — 정책명=%s", plcy_nm[:30])
        return ""


async def _enrich_policies_support(policies: list[dict]) -> list[dict]:
    """상세조회 policies에 plcySprtCnSummary 필드를 병렬로 추가한다."""
    if not policies:
        return policies
    summaries = await asyncio.gather(*[
        _summarize_support_content(p.get("plcySprtCn", ""), p.get("plcyNm", ""))
        for p in policies
    ])
    enriched = []
    for p, summary in zip(policies, summaries):
        ep = dict(p)
        if summary:
            ep["plcySprtCnSummary"] = summary
        enriched.append(ep)
    return enriched


def _shape_response(
    inquiry_types: list[str], message: str, policies: list[dict]
) -> tuple[str, list[dict]]:
    """의도(inquiry_type)에 맞춰 message/policies를 정형한다 (프론트 렌더 분기용).

    - 추천 단독: message만 보낸다 (policies 비움) — 추천 사유가 담긴 메시지 위주.
    - 상세조회 단독: policies만 보낸다 (message 비움) — 프론트가 상세 카드로 렌더.
    - 그 외(검색/비교/복합 의도/멀티 분야): 둘 다 보낸다 (정보 손실 방지).
    복합 요청("추천하고 비교")은 단독이 아니므로 둘 다 유지한다.
    """
    types = set(inquiry_types or [])
    if types == {"추천"}:
        return message, []
    if types == {"상세조회"}:
        return message, policies
    return message, policies


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
        graph.add_node("general", general_node)                     # 일반대화: 정책 무관 인사·잡담 응대
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
            "general":    "general",  # 일반대화 — 검색 우회
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
        # 일반대화는 composer 조립을 거치지 않고 바로 종료 (general_node가 final_response를 직접 채움).
        graph.add_edge("general", END)

        self.workflow = graph.compile()

    async def run(
        self,
        user_inquiry: str,
        user_role: str = "guest",
        user_profile: dict | None = None,
        messages: list | None = None, # 이전 대화 맥락 (없으면 첫 대화)
        image_base64: str | None = None,
        image_content_type: str | None = None,
        thread_id: str | None = None,  # 후속질문 정책 메모리 조회/저장 키 (없으면 메모리 미사용)
    ) -> dict:
        """챗봇 워크플로우 실행 후 프론트엔드 UI 분기에 필요한 메타까지 함께 반환.

        Returns:
            dict: {
                "message": str — composer가 조립한 최종 답변 텍스트 (상세조회 단독이면 ""),
                "category": list[str] — analysis_node가 분류한 분야 리스트 (멀티 가능),
                "inquiry_type": list[str] — 의도 리스트 (검색/추천/상세조회/비교, 복합 가능),
                "policies": list[dict] — 활성 도메인의 raw 정책 메타 (추천 단독이면 빈 리스트),
                "suggestions": list[str] — 교육 에이전트가 생성한 follow-up 질문,
            }
        """
        user_profile = user_profile or {}
        # 직전 턴 정책 메모리 조회 — 후속질문("두번째 정책"/"A와 B 비교") 해소용.
        # 메모리 생명주기(조회→주입→저장)를 supervisor가 관리해 컨트롤러는 위임만 한다.
        last_policies = get_last_policies(thread_id) if thread_id else []
        initial_state = ShareState(
            messages=messages or [],
            user_inquiry=user_inquiry,
            user_role=user_role,
            user_profile=user_profile,
            category=[],
            inquiry_type=[],
            is_general=False,
            requested_count=None,
            domain_knowledge={},
            domain_policies={},
            domain_results={},
            image_base64=image_base64,
            image_content_type=image_content_type,
            image_context="",
            last_policies=last_policies,
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
            seen: set[str] = set()
            for result in final_state.get("domain_results", {}).values():
                for p in result.get("policies", []):
                    plcy_no = p.get("plcyNo")
                    if plcy_no and plcy_no in seen:
                        continue
                    if plcy_no:
                        seen.add(plcy_no)
                    policies.append(p)

        # 이번 턴이 보여준 정책을 다음 후속질문 해소용으로 저장 (빈 결과면 내부에서 클리어).
        # 저장은 정형 전 policies로 — 추천 단독이라 응답에선 비워도 후속질문("두번째 거")은 해소돼야 한다.
        if thread_id:
            save_last_policies(thread_id, policies)

        # 의도별 응답 정형 (추천=message만 / 상세조회=policies만 / 그 외=둘 다).
        inquiry_types = final_state.get("inquiry_type", [])
        full_message = final_state["final_response"]
        message, policies = _shape_response(inquiry_types, full_message, policies)

        # 상세조회 전용: 지원 내용을 LLM으로 가독성 좋게 정리해 plcySprtCnSummary로 추가.
        # PolicyCard는 이 필드를 우선 사용하고, PolicyDetailModal은 원문(plcySprtCn)을 유지한다.
        if set(inquiry_types) == {"상세조회"} and policies:
            policies = await _enrich_policies_support(policies)

        return {
            "message": message,
            "category": final_state.get("category", []),
            "inquiry_type": inquiry_types,
            "policies": policies,
            "suggestions": final_state.get("suggestions", []),
            # 대화기록 저장용 — 상세조회처럼 message를 비워도 맥락 보존을 위해 전체 답변을 따로 넘긴다.
            # 컨트롤러가 이 값으로 히스토리를 저장하고 프론트 응답에선 제외한다.
            "full_message": full_message,
            # 이미지 분석 결과 — 컨트롤러가 대화 히스토리에 포함해 후속 턴에서 참조 가능하게 한다.
            # 프론트 응답에서는 제외 (컨트롤러가 pop).
            "image_context": final_state.get("image_context", ""),
        }

    def get_graph_image(self) -> bytes:
        return self.workflow.get_graph().draw_mermaid_png()


ChatbotSupervisorDep = Annotated[ChatbotSupervisor, Depends(ChatbotSupervisor)]