import logging
from ..state import ShareState
from ..constants import AGENT_CATEGORY, WELFARE_SIMILARITY_THRESHOLD, resolve_search_k, DEFAULT_SEARCH_K
from ..tools.policy_search import (
    vectorstore as _vectorstore,
    pick_policy_fields as _pick_policy_fields,
    similarity_to_score as _similarity_to_score,
    build_profile_query as _build_profile_query,
)
from ..tools.eligibility import is_eligibility_inquiry

logger = logging.getLogger(__name__)

_CATEGORY = AGENT_CATEGORY["welfare"]


async def welfare_search_node(state: ShareState) -> dict:
    """복지문화 정책 검색 노드.

    PGVector에서 복지 정책을 검색해 state에 기록한다. LLM은 사용하지 않는다.
    자격 검증은 하지 않는다 — 사용자가 자격을 물으면 welfare_agent가 툴을 통해 판정한다.
    """
    query = state["user_inquiry"]
    user_profile = state.get("user_profile") or {}
    inquiry_types = state.get("inquiry_type", [])
    is_recommend = "추천" in inquiry_types
    k = resolve_search_k(inquiry_types, state.get("requested_count"))

    # 자격 확인 질문은 inquiry_type이 "상세조회"여도 모든 정책 대상으로 판정해야 하므로 k 유지
    if k < DEFAULT_SEARCH_K and is_eligibility_inquiry(query):
        k = DEFAULT_SEARCH_K

    # 추천은 프로필을 검색어에 보강해 '내 상황' 적합도를 반영한다(k는 그대로 — 여전히 최대 k개).
    if is_recommend and user_profile:
        query = _build_profile_query(query, user_profile)

    logger.info("복지문화 정책 검색 노드 실행 — k=%d, query=%s", k, query[:30])

    results = await _vectorstore.asimilarity_search_with_score(
        query, k=k, filter={"category": _CATEGORY}
    )
    documents = [
        (doc, dist) for doc, dist in results if dist <= WELFARE_SIMILARITY_THRESHOLD
    ]

    if not documents:
        logger.info("복지문화 정책 검색 결과 없음")
        return {
            "domain_knowledge": {"welfare": ""},
            "domain_policies": {"welfare": []},
        }

    # 정책 메타(policies)와 knowledge 텍스트를 한 번에 조립 — 적합도 점수는 정책당 1회만 계산.
    # (추천이면 벡터 유사도를 0~100 적합도로 환산해 넣는다 — 관련도 축, 자격과 무관.)
    policies = []
    lines = []
    for idx, (doc, dist) in enumerate(documents, 1):
        score = _similarity_to_score(dist) if is_recommend else None

        p = _pick_policy_fields(doc.metadata)
        if score is not None:
            p["suitability_score"] = score
        policies.append(p)

        lines.append(f"[정책 {idx}] {doc.metadata.get('plcyNm', '')}")
        lines.append(f"내용: {doc.page_content}")
        if score is not None:
            lines.append(f"적합도: {score}점")
        lines.append(f"신청 URL: {doc.metadata.get('aplyUrlAddr', '정보 없음')}\n")
    knowledge = "\n".join(lines)

    logger.info("복지문화 정책 검색 완료 — %d건", len(policies))
    return {
        "domain_knowledge": {"welfare": knowledge},
        "domain_policies": {"welfare": policies},
    }
