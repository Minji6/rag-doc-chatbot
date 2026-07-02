import logging
from ..state import ShareState
from ..constants import AGENT_CATEGORY, SIMILARITY_DISTANCE_THRESHOLD, resolve_search_k
from ..tools.policy_search import (
    vectorstore as _vectorstore,
    pick_policy_fields as _pick_policy_fields,
    similarity_to_score as _similarity_to_score,
    build_profile_query as _build_profile_query,
)
from ..tools.eligibility import check_policy_eligibility, format_verdict_line, VERDICT_ORDER

# 로거 생성
logger = logging.getLogger(__name__)

# 카테고리 [주거]로 분류
_CATEGORY = AGENT_CATEGORY["housing"]

# 자격 진단(추천) 로직은 공통 tool로 분리됨 → from ..tools.eligibility import ... (상단)


# ============================================
# 서치 노드
# ============================================
async def housing_search_node(state: ShareState) -> dict:
    """
    주거 정책 검색 노드. LLM 미사용. 생성은 housing_node가 담당.
    추가로 사용자 프로필이 있으면 정책별 자격 진단을 수행하여
    knowledge_base에 ✅/❌/❓ 라벨을 끼워넣는다.
    '추천' 의도일 때는 벡터 유사도를 0~100 적합도(suitability_score)로 환산해 정책 메타에 넣고,
    관련도(적합도) 높은 순으로 노출한다. (적합도는 자격과 무관한 '관련도' 축)
    """
    query = state["user_inquiry"]
    user_profile = state.get("user_profile") or {}
    inquiry_types = state.get("inquiry_type", [])
    is_recommend = "추천" in inquiry_types
    k = resolve_search_k(inquiry_types, state.get("requested_count"))
    # 추천은 프로필을 검색어에 보강해 '내 상황' 적합도를 반영한다(k는 그대로 — 여전히 최대 k개).
    if is_recommend and user_profile:
        query = _build_profile_query(query, user_profile)
    logger.info("주거 정책 검색 노드 실행 - k=%d, query=%s", k, query[:30])

    results = await _vectorstore.asimilarity_search_with_score(
        query, k=k, filter={"category": _CATEGORY}
    )

    documents = [
        (doc, dist) for doc, dist in results if dist <= SIMILARITY_DISTANCE_THRESHOLD
    ]

    if not documents:
        logger.info("주거 정책 검색 결과 없음")
        return {"domain_knowledge": {"housing": ""}, "domain_policies": {"housing": []}}

    # eligibility를 한 번만 계산해 정렬·라벨 모두 재사용
    docs_with_elig = [
        (doc, dist, check_policy_eligibility(doc.metadata, user_profile))
        for doc, dist in documents
    ]
    # 추천은 관련도(적합도) 순 = 벡터 반환 순서를 유지한다. 그 외에는 기존대로 자격 적합순 정렬.
    if not is_recommend:
        docs_with_elig.sort(key=lambda item: VERDICT_ORDER[item[2]["verdict"]] if item[2] else 2)

    # 정책 메타(policies)와 knowledge 텍스트를 한 번에 조립 — 적합도 점수는 정책당 1회만 계산.
    policies = []
    lines = []
    for idx, (doc, dist, eligibility) in enumerate(docs_with_elig, 1):
        score = _similarity_to_score(dist) if is_recommend else None

        p = _pick_policy_fields(doc.metadata)
        if score is not None:
            p["suitability_score"] = score
        policies.append(p)

        lines.append(f"[정책 {idx}] {doc.metadata.get('plcyNm', '')}")
        lines.append(f"내용: {doc.page_content}")
        if eligibility is not None:
            lines.append(format_verdict_line(eligibility))
        if score is not None:
            lines.append(f"적합도: {score}점")
        lines.append(f"신청 URL: {doc.metadata.get('aplyUrlAddr', '정보 없음')}\n")
    knowledge = "\n".join(lines)

    logger.info("주거 정책 검색 완료 - %d건", len(policies))
    return {
        "domain_knowledge": {"housing": knowledge},
        "domain_policies": {"housing": policies},
    }
