# api/chat_service/langgraph/supervisor.py
import json
import logging
from functools import lru_cache
from typing import Annotated, Hashable, cast
from fastapi import Depends
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
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

# ── 지원 내용 배치 요약(LLM) ──────────────────────────────────────────────────
# 상세조회 policies의 '지원 내용' 원문을 정책당 LLM 호출하면 N건 = N번 호출이 되어
# 비용이 커진다. composer의 비교 표 셀 요약(_summarize_compare_cells)과 동일하게
# 구조화 출력(with_structured_output)으로 한 번의 배치 호출에 담아 처리한다.

class _SupportSummary(BaseModel):
    """정책 1건의 지원 내용 요약."""
    idx: Annotated[int, Field(description="입력 정책의 0-based 순번")]
    summary: Annotated[str, Field(description="2~3문장 요약 (없으면 빈 문자열)")] = ""


class _SupportSummaries(BaseModel):
    """정책들의 지원 내용 요약 묶음."""
    items: list[_SupportSummary]


_support_summary_model = init_chat_model(
    "gpt-4o-mini", model_provider="openai", temperature=0.0
).with_structured_output(_SupportSummaries)

_ENRICH_SYSTEM_PROMPT = """당신은 청년정책 안내문 요약 AI입니다.
각 정책의 '지원 내용' 원문을 2~3문장의 짧은 요약문으로 작성하세요.

규칙:
- 2~3문장 이내로 핵심(지원 금액·기간·지원 방식·주요 조건)만 남기세요.
- 세부 절차나 부가 조건은 생략해도 됩니다.
- 존댓말(~합니다/~됩니다)을 사용하세요.
- 서식(제목·불릿·번호·마크다운) 없이 평문으로만 쓰세요.
- "신청하세요", "활용해 보세요" 같은 권유·홍보 문구는 쓰지 마세요.
- 원문에 없는 정보를 추가하거나 지어내지 마세요.
- 입력으로 받은 모든 정책을 idx 그대로 매겨 빠짐없이 반환하세요."""


async def _enrich_policies_support(policies: list[dict]) -> list[dict]:
    """상세조회 policies에 plcySprtCnSummary 필드를 배치 LLM 호출로 추가한다."""
    if not policies:
        return policies

    items = []
    for idx, p in enumerate(policies):
        plcy_sprt_cn = p.get("plcySprtCn", "")
        if not plcy_sprt_cn or not str(plcy_sprt_cn).strip():
            continue
        items.append({
            "idx": idx,
            "plcyNm": p.get("plcyNm", ""),
            "plcySprtCn": plcy_sprt_cn,
        })

    if not items:
        return [dict(p) for p in policies]

    try:
        result = cast(_SupportSummaries, await _support_summary_model.ainvoke([
            SystemMessage(content=_ENRICH_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(items, ensure_ascii=False)),
        ]))
    except Exception:
        logger.exception("지원내용 LLM 배치 요약 실패 — %d건", len(items))
        return [dict(p) for p in policies]

    summaries_by_idx: dict[int, str] = {}
    for entry in result.items:
        if not (0 <= entry.idx < len(policies)):
            continue
        if entry.summary.strip():
            summaries_by_idx[entry.idx] = entry.summary.strip()

    enriched = []
    for idx, p in enumerate(policies):
        ep = dict(p)
        summary = summaries_by_idx.get(idx)
        if summary:
            ep["plcySprtCnSummary"] = summary
        enriched.append(ep)
    return enriched


def _shape_response(
    inquiry_types: list[str], message: str, policies: list[dict]
) -> tuple[str, list[dict]]:
    """의도(inquiry_type)에 맞춰 message/policies를 정형한다 (프론트 렌더 분기용).

    현재는 모든 의도가 message·policies를 함께 반환한다.
    추천 단독일 때는 policies 각 항목에 suitability_score(적합도, 0~100 관련도 점수)가 담겨 있어
    프론트가 뱃지·정렬 등에 구조화된 값으로 활용할 수 있다
    (과거엔 추천 단독이면 policies를 비웠으나, 적합도를 구조화 필드로 내보내기 위해 유지한다).
    """
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
                "policies": list[dict] — 활성 도메인의 raw 정책 메타 (추천이면 정책별 suitability_score 포함),
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


@lru_cache(maxsize=1)
def get_chatbot_supervisor() -> ChatbotSupervisor:
    return ChatbotSupervisor()


ChatbotSupervisorDep = Annotated[ChatbotSupervisor, Depends(get_chatbot_supervisor)]