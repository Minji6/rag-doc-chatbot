import logging
from ..state import ShareState
from ..tools.policy_search import (
    vectorstore as _vectorstore,
    pick_policy_fields as _pick_policy_fields,
    similarity_to_score as _similarity_to_score,
    build_profile_query as _build_profile_query,
)
from ..tools.eligibility import check_policy_eligibility, format_verdict_line, VERDICT_ORDER
from ..constants import (
    AGENT_CATEGORY,
    ROLE_GUEST,
    EDUCATION_SIMILARITY_THRESHOLD_USER,
    EDUCATION_SIMILARITY_THRESHOLD_GUEST,
    resolve_search_k,
)

logger = logging.getLogger(__name__)

_CATEGORY = AGENT_CATEGORY["education"]


async def education_search_node(state: ShareState) -> dict:
    """교육 정책 검색 노드.

    PGVector에서 교육 정책을 검색해 state에 기록한다. LLM은 사용하지 않는다.
    (검색/생성 분리 구조 — 생성은 education_node가 담당)

    inquiry_type별 검색 전략:
    - 추천: 유저 프로필을 쿼리에 덧붙여 벡터 검색 품질 향상 (guest면 일반 검색과 동일)
    - 상세조회: k=2로 줄여 가장 유사한 정책에 집중
    - 검색/비교: 기본 k=5

    유저 역할별 임계값:
    - guest: _SIMILARITY_THRESHOLD_GUEST (완화) — 정확도보다 결과 제공 우선
    - user: _SIMILARITY_THRESHOLD_USER — 관련성 있는 정책 위주

    - domain_knowledge["education"]: 생성 노드가 답변 근거로 읽을 정책 텍스트
    - domain_policies["education"]: composer 후처리(비교/점수)용 raw 정책 메타
    """
    inquiry = state["user_inquiry"]
    inquiry_types = state.get("inquiry_type", [])
    is_recommend = "추천" in inquiry_types
    user_role = state.get("user_role", ROLE_GUEST)
    user_profile = state.get("user_profile") or {}

    threshold = EDUCATION_SIMILARITY_THRESHOLD_GUEST if user_role == ROLE_GUEST else EDUCATION_SIMILARITY_THRESHOLD_USER

    # 검색 정책 수(k)는 4개 도메인 공통(resolve_search_k). 교육은 추천 시 프로필을
    # 쿼리에 보강하는 전략만 별도로 유지한다.
    k = resolve_search_k(inquiry_types, state.get("requested_count"))
    if is_recommend and user_profile:
        query = _build_profile_query(inquiry, user_profile)
    else:  # 검색, 상세조회, 비교
        query = inquiry

    logger.info("교육 정책 검색 노드 실행 — inquiry_type=%s, user_role=%s, query=%s",
                inquiry_types, user_role, query[:40])

    results = await _vectorstore.asimilarity_search_with_score(
        query, k=k, filter={"category": _CATEGORY}
    )
    documents = [
        (doc, dist) for doc, dist in results if dist <= threshold
    ]

    if not documents:
        logger.info("교육 정책 검색 결과 없음")
        return {
            "domain_knowledge": {"education": ""},
            "domain_policies": {"education": []},
        }

    # eligibility를 한 번만 계산해 정렬·라벨 모두 재사용
    docs_with_elig = [
        (doc, dist, check_policy_eligibility(doc.metadata, user_profile))
        for doc, dist in documents
    ]
    # 추천은 관련도(적합도) 순 = 벡터 반환 순서를 유지한다. 그 외에는 기존대로 자격 적합순 정렬.
    if not is_recommend:
        docs_with_elig.sort(key=lambda item: VERDICT_ORDER[item[2]["verdict"]] if item[2] else 2)

    policies = []
    for doc, dist, _elig in docs_with_elig:
        p = _pick_policy_fields(doc.metadata)
        if is_recommend:
            p["suitability_score"] = _similarity_to_score(dist)
        policies.append(p)

    lines = []
    for idx, (doc, dist, eligibility) in enumerate(docs_with_elig, 1):
        lines.append(f"[정책 {idx}] {doc.metadata.get('plcyNm', '')}")
        lines.append(f"내용: {doc.page_content}")

        if eligibility is not None:
            lines.append(format_verdict_line(eligibility))
        if is_recommend:
            lines.append(f"적합도: {_similarity_to_score(dist)}점")

        lines.append(f"신청 URL: {doc.metadata.get('aplyUrlAddr', '정보 없음')}\n")
    knowledge = "\n".join(lines)

    logger.info("교육 정책 검색 완료 — %d건 (threshold=%.1f)", len(policies), threshold)
    return {
        "domain_knowledge": {"education": knowledge},
        "domain_policies": {"education": policies},
    }
