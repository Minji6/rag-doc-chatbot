import logging
from ..state import ShareState
from ..constants import AGENT_CATEGORY, SIMILARITY_DISTANCE_THRESHOLD, resolve_search_k
from ..tools.policy_search import vectorstore as _vectorstore, pick_policy_fields as _pick_policy_fields
from ..tools.eligibility import (
    check_policy_eligibility,
    format_verdict_line,
    eligibility_sort_key,
)

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
    """
    query = state["user_inquiry"]
    user_profile = state.get("user_profile") or {}
    k = resolve_search_k(state.get("inquiry_type", []), state.get("requested_count"))
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

    # 자격 진단 결과 기준으로 정렬: eligible → unknown → ineligible (게스트는 검색 순서 유지)
    documents.sort(key=lambda item: eligibility_sort_key(item[0].metadata, user_profile))

    policies = [_pick_policy_fields(doc.metadata) for doc, _ in documents]

    lines = []
    for idx, (doc, _dist) in enumerate(documents, 1):
        lines.append(f"[정책 {idx}] {doc.metadata.get('plcyNm', '')}")
        lines.append(f"내용: {doc.page_content}")

        # 자격 진단 라벨 (로그인 유저만, 게스트는 None 반환되어 스킵)
        eligibility = check_policy_eligibility(doc.metadata, user_profile)
        if eligibility is not None:
            lines.append(format_verdict_line(eligibility))

        lines.append(f"신청 URL: {doc.metadata.get('aplyUrlAddr', '정보 없음')}\n")
    knowledge = "\n".join(lines)

    logger.info("주거 정책 검색 완료 - %d건", len(policies))
    return {
        "domain_knowledge": {"housing": knowledge},
        "domain_policies": {"housing": policies},
    }
