from typing import Annotated, Literal, TypedDict
from langgraph.graph import add_messages


class DomainResult(TypedDict):
    """도메인 에이전트(주거/취업/교육/복지) 표준 반환 스키마.

    - text: 도메인 에이전트가 만든 본문 조각(### 정책명 블록들). composer가 조립.
    - policies: search_policy가 PGVector에서 찾은 raw 정책 메타(화이트리스트 적용)
                후속 비교/점수 처리용 raw 데이터
    - category: 어느 분야 결과인지 (멀티 분야 병합 시 출처 구분용)
    - source: 정책 출처 — "rag"(PGVector) / "web"(웹 보완) / "none"(결과 없음)
    """
    text: str
    policies: list[dict]
    category: str
    source: Literal["rag", "web", "none"]


def empty_domain_result(category: str = "") -> DomainResult:
    """초기 state 및 fallback에 쓰는 빈 DomainResult."""
    return DomainResult(text="", policies=[], category=category, source="none")


def _merge_dict(left: dict | None, right: dict | None) -> dict:
    """LangGraph reducer: 두 dict를 얕게 머지.

    같은 키가 오면 right(나중에 들어온 쪽)가 이김.
    fan-out 시 도메인별 키만 들어오므로 충돌은 정상 동작에선 일어나지 않음.
    None/빈 값에 안전.
    """
    if not left:
        return dict(right) if right else {}
    if not right:
        return dict(left)
    return {**left, **right}


class ShareState(TypedDict):
    messages: Annotated[list, add_messages]   # 필수 — 대화 누적

    user_inquiry: str      # 사용자 원본 질문
    user_role: str         # "guest" 또는 "user" (로그인 여부)
    user_profile: dict     # 로그인 유저 프로필 (guest일 경우 빈 dict)

    category: list[str]    # 분야 리스트: ["주거", "일자리", ...] — 멀티 분야 라우팅용
    inquiry_type: str      # 의도: 검색 / 추천 / 상세조회 / 비교

    # 검색·생성 핸드오프 + 결과 채널 — 도메인 이름(housing/employment/education/welfare)을 키로 가진 dict.
    # fan-out 시 각 도메인이 자기 키에만 쓰고, _merge_dict reducer가 자동 머지한다.
    domain_knowledge: Annotated[dict[str, str], _merge_dict]        # search_node → domain_node 핸드오프
    domain_policies: Annotated[dict[str, list[dict]], _merge_dict]  # raw 정책 메타 (composer 후처리용)
    domain_results: Annotated[dict[str, DomainResult], _merge_dict] # 도메인별 본문 조각(fragment)

    image_base64: str | None       # 사용자가 첨부한 이미지 (base64 인코딩)
    image_content_type: str | None # 이미지 MIME 타입 (예: "image/jpeg")
    image_context: str             # image_analysis_node가 추출한 이미지 정보 (없으면 빈 문자열)

    suggestions: list[str]     # 교육 에이전트가 생성한 follow-up 질문 (PoC)
    final_response: str        # composer_node가 채우는 최종 답변
