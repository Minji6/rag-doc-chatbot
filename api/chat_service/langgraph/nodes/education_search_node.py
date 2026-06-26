import logging
from ..state import ShareState
from ..tools.policy_search import vectorstore as _vectorstore, pick_policy_fields as _pick_policy_fields
from ..constants import (
    AGENT_CATEGORY,
    ROLE_GUEST,
    EDUCATION_SIMILARITY_THRESHOLD_USER,
    EDUCATION_SIMILARITY_THRESHOLD_GUEST,
    resolve_search_k,
)

logger = logging.getLogger(__name__)

_CATEGORY = AGENT_CATEGORY["education"]


def _build_profile_query(inquiry: str, user_profile: dict) -> str:
    """추천 의도일 때 유저 프로필을 쿼리에 덧붙여 벡터 검색 품질을 높인다.

    실제 user_profile 키(auth_service/model.py 기준):
    birth_date, zipcd, schoolcd, plcymajorcd, jobcd, earncndsecd, mrgsttscd, sbizcd

    '제한없음'은 유효한 값으로 그대로 포함한다 — 제한 없는 정책과 매칭되도록.
    None/빈값만 제외한다.
    """
    profile_parts = []
    if user_profile.get("birth_date"):
        profile_parts.append(f"생년월일:{user_profile['birth_date']}")
    if user_profile.get("zipcd"):
        profile_parts.append(f"거주지역:{user_profile['zipcd']}")
    if user_profile.get("schoolcd"):
        profile_parts.append(f"학교:{user_profile['schoolcd']}")
    if user_profile.get("plcymajorcd"):
        profile_parts.append(f"전공:{user_profile['plcymajorcd']}")
    if user_profile.get("jobcd"):
        profile_parts.append(f"직업:{user_profile['jobcd']}")
    if user_profile.get("earncndsecd") is not None:
        profile_parts.append(f"월소득:{user_profile['earncndsecd']}원")
    if user_profile.get("mrgsttscd"):
        profile_parts.append(f"혼인상태:{user_profile['mrgsttscd']}")
    if user_profile.get("sbizcd"):
        profile_parts.append(f"소상공인업종:{user_profile['sbizcd']}")

    return f"{inquiry} / {' '.join(profile_parts)}" if profile_parts else inquiry


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
    inquiry_type = state.get("inquiry_type", "검색")
    user_role = state.get("user_role", ROLE_GUEST)
    user_profile = state.get("user_profile") or {}

    threshold = EDUCATION_SIMILARITY_THRESHOLD_GUEST if user_role == ROLE_GUEST else EDUCATION_SIMILARITY_THRESHOLD_USER

    # 검색 정책 수(k)는 4개 도메인 공통(resolve_search_k). 교육은 추천 시 프로필을
    # 쿼리에 보강하는 전략만 별도로 유지한다.
    k = resolve_search_k(inquiry_type)
    if inquiry_type == "추천" and user_profile:
        query = _build_profile_query(inquiry, user_profile)
    else:  # 검색, 상세조회, 비교
        query = inquiry

    logger.info("교육 정책 검색 노드 실행 — inquiry_type=%s, user_role=%s, query=%s",
                inquiry_type, user_role, query[:40])

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

    policies = [_pick_policy_fields(doc.metadata) for doc, _ in documents]

    lines = []
    for idx, (doc, _dist) in enumerate(documents, 1):
        lines.append(f"[정책 {idx}] {doc.metadata.get('plcyNm', '')}")
        lines.append(f"내용: {doc.page_content}")
        lines.append(f"신청 URL: {doc.metadata.get('aplyUrlAddr', '정보 없음')}\n")
    knowledge = "\n".join(lines)

    logger.info("교육 정책 검색 완료 — %d건 (threshold=%.1f)", len(policies), threshold)
    return {
        "domain_knowledge": {"education": knowledge},
        "domain_policies": {"education": policies},
    }
