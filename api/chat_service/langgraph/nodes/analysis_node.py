import logging
import re
from typing import Annotated, cast
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from ..state import ShareState
from ..constants import CATEGORIES, INQUIRY_TYPES
from ..tools.eligibility import is_eligibility_inquiry

logger = logging.getLogger(__name__)

# 추천/검색(목록 요청) 의도를 드러내는 단어 — 자격질문 보정에서 제외하기 위한 안전장치.
# ("받을 수 있는 정책 추천해줘"처럼 자격 표현이 섞여도 목록 요청이면 상세조회로 강제하지 않음)
_RECOMMEND_LIST_RE = re.compile(r"추천|알려줘|찾아|뭐\s*있|어떤\s*(게|거|것)|목록|리스트")


def _is_specific_eligibility(inquiry: str) -> bool:
    """특정 정책의 신청 가능/자격 여부만 묻는(목록 요청 아님) 질문인지 판정."""
    return is_eligibility_inquiry(inquiry) and not _RECOMMEND_LIST_RE.search(inquiry)


class InquiryAnalysis(BaseModel):
    """사용자 질문의 분야·의도 분류 결과 구조화 출력 스키마"""
    is_general: Annotated[
        bool,
        Field(
            description=(
                "정책과 무관한 인사·잡담·일반 질문(예: '안녕', '오늘 날씨', '파이썬이 뭐야')이면 true, "
                "청년정책(일자리·주거·교육·복지문화) 관련 질문이면 false. "
                "true면 category와 inquiry_type은 빈 리스트로 둔다."
            )
        )
    ]
    category: Annotated[
        list[str],
        Field(
            description=(
                f"정책 분야 리스트. 다음 중에서만 선택: {', '.join(CATEGORIES)}. "
                f"질문이 한 분야면 1개, 여러 분야에 걸치면 모두 나열. is_general=true면 빈 리스트."
            )
        )
    ]
    inquiry_type: Annotated[
        list[str],
        Field(
            description=(
                f"질문 의도 리스트. 다음 중에서만 선택: {', '.join(INQUIRY_TYPES)}. "
                f"보통 1개지만 '추천하고 비교해줘'처럼 둘 이상이면 모두 담는다. is_general=true면 빈 리스트."
            )
        )
    ]
    requested_count: Annotated[
        int | None,
        Field(
            description=(
                "사용자가 명시적으로 요청한 정책 개수. '3개', '다섯 개', '세 가지'처럼 "
                "수가 분명히 적힌 경우에만 정수로, 개수 언급이 없으면 null."
            )
        )
    ] = None


# 모듈 싱글톤 — 한 번만 생성 (지침서 22-2)
_chat_model = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0.0)
_structured_model = _chat_model.with_structured_output(InquiryAnalysis)

_SYSTEM_PROMPT = f"""당신은 청년정책 챗봇의 의도 분석기입니다.
사용자 질문을 아래 두 기준으로 분류하세요.

[분야 - category] {', '.join(CATEGORIES)} 중에서 선택 (복수 가능)
- 일자리: 취업·창업·재직자 지원 등
- 주거: 전세대출·월세지원·공공주택 등
- 교육: 장학금·학자금·내일배움카드 등
- 복지문화: 금융지원·문화예술·복지 등

[일반대화 - is_general] 다음 중 하나면 true. is_general=true면 category·inquiry_type은 빈 리스트로 둔다.
- 인사·잡담·일반 질문: "안녕", "고마워", "오늘 기분 어때", "파이썬이 뭐야" 등
- 이미지·대화에서 이미 추출된 수치를 확인하는 단순 사실 질문 (정책 요청 없이 수치만 묻는 경우):
  "내 학점이 몇이야?", "학점이 몇인데?" 등 → true
  ※ 정책 요청이 한 문장이라도 결합되면 반드시 false:
     "학점 기준으로 정책 추천해줘" → false
     "소득분위가 몇이야? 국가장학금 받을 수 있어?" → false (정책 가능 여부 포함)
     "내 상황에 맞는 정책 알려줘" → false

분류 규칙:
- 한 분야에 해당하면 그 분야만 리스트에 담아 반환 (예: ["주거"])
- 여러 분야에 걸치면 해당하는 분야를 모두 리스트에 담아 반환 (예: ["주거", "일자리"])
- 명확히 한 분야가 떠오르면 그 하나만 선택. 모호할 때만 복수 선택.
- **비교 의도일 때**: 비교 대상 정책들이 서로 다른 분야에 속하면 **모든 분야를 빠짐없이** 담아야 한다.
  예: "청년 월세 지원이랑 청년내일채움공제 비교해줘" → ["주거", "일자리"] (두 분야 모두 필수)

[의도 - inquiry_type] {', '.join(INQUIRY_TYPES)} 중 하나 이상 (리스트). **기본값은 '검색'.**
- 검색: 분야·키워드로 정책 목록을 찾고 싶어함. **대부분의 "~정책 알려줘 / 찾아줘 / 뭐 있어? / 같이 알려줘"는 검색이다.**
  예: "청년 월세 지원 정책 알려줘", "주거랑 일자리 정책 같이 알려줘", "전세대출 뭐 있어?" → ["검색"]
- 추천: 사용자가 **본인 상황·조건을 설명**하거나 "추천/맞는/적합한/받을만한"을 명시할 때만.
  예: "사회초년생인데 받을만한 거 추천해줘", "내 소득에 맞는 정책" → ["추천"]
- 상세조회: **특정 정책 1개를 고유 명칭으로 콕 집어** 그 정책의 세부(자격·금액·절차 등)를 물을 때만.
  예: "청년도약계좌 자세히 알려줘", "행복주택 신청조건이 뭐야?" → ["상세조회"]
  ※ 분야·키워드 수준의 "~정책 알려줘"는 상세조회가 아니라 **검색**이다.
  ※ **특정 정책에 대해 본인의 신청 가능 여부·자격 여부를 묻는 질문은 모두 상세조회다.** 절대 추천으로 분류하지 말 것.
    예: "내가 이 정책 신청할 자격이 돼?", "신청할 수 있어?", "받을 수 있어?", "대상이야?",
        "자격이 되나?", "이 정책 지원 가능한가?" → ["상세조회"]
- 비교: 둘 이상 정책의 차이를 묻거나 "비교해줘".
- "추천하고 비교해줘"처럼 둘 이상이면 모두 담는다 (예: ["추천", "비교"]).

[개수 - requested_count] "3개", "다섯 개"처럼 개수가 명시되면 정수로, 없으면 null.

분야가 모호하면 가장 가까운 값으로 분류하되, 의도가 애매하면 '검색'을 택하세요."""


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
    # 후속 정책질문(직전 정책 특정)은 일반대화가 아니라 정책 처리이므로 is_general 강제 해제.
    is_general = bool(analysis.is_general) and not resolved
    if resolved:
        seen: list[str] = []
        for p in resolved:
            c = (p.get("category") or "").strip()
            if c and c not in seen:
                seen.append(c)
        # 비교 의도일 때는 LLM이 분류한 category도 병합한다.
        # resolved_policies가 한 분야만 커버해도 새로 언급된 타 분야 정책을
        # 놓치지 않기 위함 (예: last_policies에 주거만 있는데
        # "이거랑 청년내일채움공제 비교해줘" → 일자리도 검색돼야 함).
        if "비교" in (analysis.inquiry_type or []):
            for c in (analysis.category or []):
                if c and c not in seen:
                    seen.append(c)
        categories = seen or (analysis.category or [])
        logger.info("resolved_policies로 분야 확정 — category=%s", categories)
    else:
        # LLM이 빈 리스트를 반환하는 케이스 방어 — 라우팅 fallback이 받아 처리
        categories = analysis.category or []

    inquiry_types = analysis.inquiry_type or []

    # 결정적 보정 — 특정 정책의 자격 여부 질문("신청할 수 있어?", "자격 돼?")은
    # LLM이 간혹 '추천/검색'으로 흘리므로, 목록 요청이 아닌 자격질문이면 상세조회를 확정한다.
    # 단 "행복주택 자격 돼? 청년버팀목이랑 비교도"처럼 함께 온 '비교' 의도는 보존한다
    # (추천/검색은 특정 정책 자격질문에서 오분류이므로 버린다).
    if not is_general and _is_specific_eligibility(state["user_inquiry"]) and "상세조회" not in inquiry_types:
        logger.info("자격 여부 질문 감지 — inquiry_type을 상세조회로 보정 (원본=%s)", inquiry_types)
        inquiry_types = (["비교"] if "비교" in inquiry_types else []) + ["상세조회"]

    logger.info(
        "분류 결과 — is_general=%s, category=%s, inquiry_type=%s, requested_count=%s",
        is_general, categories, inquiry_types, analysis.requested_count,
    )
    return {
        "category": categories,
        "inquiry_type": inquiry_types,
        "is_general": is_general,
        "requested_count": analysis.requested_count,
    }