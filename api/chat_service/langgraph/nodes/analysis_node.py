import logging
from typing import Annotated, cast
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from ..state import ShareState
from ..constants import CATEGORIES, INQUIRY_TYPES

logger = logging.getLogger(__name__)


class InquiryAnalysis(BaseModel):
    """사용자 질문의 분야·의도 분류 결과 구조화 출력 스키마"""
    category: Annotated[
        list[str],
        Field(
            description=(
                f"정책 분야 리스트. 다음 중에서만 선택: {', '.join(CATEGORIES)}. "
                f"질문이 한 분야면 1개, 여러 분야에 걸치면 모두 나열."
            )
        )
    ]
    inquiry_type: Annotated[
        str,
        Field(description=f"질문 의도. 다음 중 하나만 사용: {', '.join(INQUIRY_TYPES)}")
    ]


# 모듈 싱글톤 — 한 번만 생성 (지침서 22-2)
_chat_model = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0.0)
_structured_model = _chat_model.with_structured_output(InquiryAnalysis)

_SYSTEM_PROMPT = f"""당신은 청년정책 챗봇의 의도 분석기입니다.
사용자 질문을 아래 두 기준으로 분류하세요.

[분야 - category] {', '.join(CATEGORIES)} 중에서 선택 (복수 가능)
- 일자리: 취업·창업·재직자 지원 등
- 주거: 전세대출·월세지원·공공주택 등
- 교육: 장학금·학자금·내일배움카드 등. 그리고 **소득분위 추정·계산, 가구 소득으로 소득분위 판단,
  학점(GPA) 진단, 교육·훈련의 급여/비급여(훈련비 지원) 여부 판단**도 교육으로 분류한다.
  (이 계산 도구들이 교육 분야에 있으므로, 소득분위·학점·훈련비 계산 요청은 교육으로 보낸다.)
- 복지문화: 금융지원·문화예술·복지 등

분류 규칙:
- 한 분야에 해당하면 그 분야만 리스트에 담아 반환 (예: ["주거"])
- 여러 분야에 걸치면 해당하는 분야를 모두 리스트에 담아 반환 (예: ["주거", "일자리"])
- 명확히 한 분야가 떠오르면 그 하나만 선택. 모호할 때만 복수 선택.
- **소득분위/소득 계산·학점 진단·훈련비(급여/비급여) 질문은 다른 분야 키워드가 없으면 ["교육"]으로 분류.**
  (예: "월 60만원 4인가구 소득분위 알려줘" → ["교육"], "내 학점으로 받을 수 있는 거" → ["교육"])

[의도 - inquiry_type] {', '.join(INQUIRY_TYPES)} 중 하나
- 검색: 특정 키워드로 정책을 찾고 싶어함
- 추천: 본인 상황을 설명하며 맞는 정책을 추천받고 싶어함
- 상세조회: 특정 정책의 세부 내용을 물어봄
- 비교: 여러 정책을 비교해달라고 함

애매하면 가장 가까운 값으로 분류하세요."""


async def analysis_node(state: ShareState) -> dict:
    profile = state["user_profile"]
    logger.info("의도 분석 노드 실행 — user_role=%s, user_profile=%s",
                state["user_role"], f"user_id={profile['user_id']}" if profile else "없음")

    user_content = state["user_inquiry"]
    image_context = state.get("image_context", "")
    if image_context:
        user_content = f"{user_content}\n\n[첨부 이미지 내용]\n{image_context}"

    analysis = cast(InquiryAnalysis, await _structured_model.ainvoke([
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]))

    # 후속질문이 직전 정책을 특정했으면(resolved_policies) 분야를 그 정책들의 분야로 확정한다.
    # → "두번째 정책 자세히"의 재작성 질의에 '근로자' 같은 단어가 섞여 LLM이 엉뚱한 분야를
    #   추가 분류하는 것을 차단(라우팅이 지정 정책의 분야로만 가도록). 의도(inquiry_type)는 그대로 사용.
    resolved = state.get("resolved_policies") or []
    if resolved:
        seen: list[str] = []
        for p in resolved:
            c = (p.get("category") or "").strip()
            if c and c not in seen:
                seen.append(c)
        categories = seen or (analysis.category or [])
        logger.info("resolved_policies로 분야 확정 — category=%s", categories)
    else:
        # LLM이 빈 리스트를 반환하는 케이스 방어 — 라우팅 fallback이 받아 처리
        categories = analysis.category or []

    logger.info("분류 결과 — category=%s, inquiry_type=%s", categories, analysis.inquiry_type)
    return {"category": categories, "inquiry_type": analysis.inquiry_type}