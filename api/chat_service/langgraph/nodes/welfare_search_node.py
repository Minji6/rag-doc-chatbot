import logging
from ..state import ShareState
from ..constants import AGENT_CATEGORY, WELFARE_SIMILARITY_THRESHOLD, resolve_search_k
from ..tools.policy_search import vectorstore as _vectorstore, pick_policy_fields as _pick_policy_fields
from ..tools.eligibility import check_policy_eligibility, format_verdict_line, VERDICT_ORDER

logger = logging.getLogger(__name__)

_CATEGORY = AGENT_CATEGORY["welfare"]


async def welfare_search_node(state: ShareState) -> dict:
    """복지문화 정책 검색 노드.

    PGVector에서 복지 정책을 검색해 state에 기록한다. LLM은 사용하지 않는다.
    (검색/생성 분리 구조 — 생성은 welfare_node가 담당)

    - welfare_knowledge_base: 생성 노드가 답변 근거로 읽을 정책 텍스트
    - welfare_policies: composer 후처리(비교/점수)용 raw 정책 메타
    """
    query = state["user_inquiry"]
    user_profile = state.get("user_profile") or {}
    k = resolve_search_k(state.get("inquiry_type", []), state.get("requested_count"))
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

    # eligibility를 한 번만 계산해 정렬·라벨 모두 재사용
    docs_with_elig = [
        (doc, dist, check_policy_eligibility(doc.metadata, user_profile))
        for doc, dist in documents
    ]
    docs_with_elig.sort(key=lambda item: VERDICT_ORDER[item[2]["verdict"]] if item[2] else 2)

    policies = [_pick_policy_fields(doc.metadata) for doc, _, _ in docs_with_elig]

    lines = []
    for idx, (doc, _dist, eligibility) in enumerate(docs_with_elig, 1):
        lines.append(f"[정책 {idx}] {doc.metadata.get('plcyNm', '')}")
        lines.append(f"내용: {doc.page_content}")

        if eligibility is not None:
            lines.append(format_verdict_line(eligibility))

        lines.append(f"신청 URL: {doc.metadata.get('aplyUrlAddr', '정보 없음')}\n")
    knowledge = "\n".join(lines)

    logger.info("복지문화 정책 검색 완료 — %d건", len(policies))
    return {
        "domain_knowledge": {"welfare": knowledge},
        "domain_policies": {"welfare": policies},
    }
