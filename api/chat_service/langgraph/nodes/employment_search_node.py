import logging
from datetime import date
from ..state import ShareState
from ..tools.dday import end_date_from
from ..tools.policy_search import (
    vectorstore as _vectorstore,
    pick_policy_fields as _pick_policy_fields,
    similarity_to_score as _similarity_to_score,
    build_profile_query as _build_profile_query,
)
from ..tools.eligibility import check_policy_eligibility, format_verdict_line, VERDICT_ORDER
from ..constants import AGENT_CATEGORY, EMPLOYMENT_SIMILARITY_THRESHOLD, resolve_search_k, EMPLOYMENT_SEARCH_OVERSAMPLE

logger = logging.getLogger(__name__)

_CATEGORY = AGENT_CATEGORY["employment"]


def _dday_label(aply_prd_se_cd: str, aply_ymd: str) -> str | None:
    """신청기간 구분 + 날짜 문자열로 D-day 라벨을 반환한다.
    반환값이 None이면 마감/만료 → 호출부에서 정책을 제외한다.
    """
    se = (aply_prd_se_cd or "").strip()
    if se == "마감":
        return None
    if se == "상시":
        return "상시접수"

    # 특정기간: 공통 파서로 aplyYmd에서 마감일(끝 날짜)을 구한다.
    end_date = end_date_from(apply_ymd=aply_ymd)
    if end_date is None:
        return ""  # 날짜 없음/파싱 실패 → 라벨 없이 통과 (safe default)

    delta = (end_date - date.today()).days

    if delta < 0:
        return f"⚠️ 현재 신청 불가 (신청기간 {end_date.strftime('%Y.%m.%d')} 종료)"
    if delta == 0:
        return f"D-day (오늘 마감 {end_date.strftime('%Y.%m.%d')})"
    return f"D-{delta} ({end_date.strftime('%Y.%m.%d')} 마감)"


async def employment_search_node(state: ShareState) -> dict:
    """취업 정책 검색 노드.

    PGVector에서 취업 정책을 검색해 state에 기록한다. LLM은 사용하지 않는다.
    (검색/생성 분리 구조 — 생성은 employment_node가 담당)

    - domain_knowledge["employment"]: 생성 노드가 답변 근거로 읽을 정책 텍스트
    - domain_policies["employment"]: composer 후처리(비교/점수)용 raw 정책 메타
    """
    query = state["user_inquiry"]
    user_profile = state.get("user_profile") or {}
    inquiry_types = state.get("inquiry_type", [])
    is_recommend = "추천" in inquiry_types
    # 출력 정책 수(k)는 4개 도메인 공통(resolve_search_k). 단 취업은 만료 정책을 걸러내므로
    # 후보를 k배(oversample)만큼 더 가져온 뒤, 살아남은 것 중 k개만 사용한다.
    k = resolve_search_k(inquiry_types, state.get("requested_count"))
    fetch_k = k * EMPLOYMENT_SEARCH_OVERSAMPLE
    # 추천은 프로필을 검색어에 보강해 '내 상황' 적합도를 반영한다(k는 그대로 — 여전히 최대 k개).
    if is_recommend and user_profile:
        query = _build_profile_query(query, user_profile)
    logger.info("취업 정책 검색 노드 실행 — k=%d(후보 %d), query=%s", k, fetch_k, query[:30])

    results = await _vectorstore.asimilarity_search_with_score(
        query, k=fetch_k, filter={"category": _CATEGORY}
    )

    documents_with_dday: list[tuple] = []
    for doc, dist in results:
        if dist > EMPLOYMENT_SIMILARITY_THRESHOLD:
            continue
        label = _dday_label(
            doc.metadata.get("aplyPrdSeCd", ""),
            doc.metadata.get("aplyYmd", ""),
        )
        if label is None:  # 마감 / 기간 만료
            continue
        documents_with_dday.append((doc, dist, label))

    # eligibility를 한 번만 계산해 정렬·라벨 모두 재사용
    docs_with_elig: list[tuple] = [
        (doc, dist, label, check_policy_eligibility(doc.metadata, user_profile))
        for doc, dist, label in documents_with_dday
    ]
    # 추천은 관련도(적합도) 순 = 벡터 반환 순서를 유지한다. 그 외에는 기존대로 자격 적합순 정렬.
    if not is_recommend:
        docs_with_elig.sort(key=lambda item: VERDICT_ORDER[item[3]["verdict"]] if item[3] else 2)
    docs_with_elig = docs_with_elig[:k]  # 출력은 공통 k개로 제한

    if not docs_with_elig:
        logger.info("취업 정책 검색 결과 없음")
        return {
            "domain_knowledge": {"employment": ""},
            "domain_policies": {"employment": []},
        }

    # 정책 메타(policies)와 knowledge 텍스트를 한 번에 조립 — 적합도 점수는 정책당 1회만 계산.
    policies = []
    lines = []
    for idx, (doc, dist, dday, eligibility) in enumerate(docs_with_elig, 1):
        score = _similarity_to_score(dist) if is_recommend else None

        p = _pick_policy_fields(doc.metadata)
        if score is not None:
            p["suitability_score"] = score
        policies.append(p)

        lines.append(f"[정책 {idx}] {doc.metadata.get('plcyNm', '')}")
        lines.append(f"내용: {doc.page_content}")
        if dday:
            lines.append(f"신청기간: {dday}")
        if eligibility is not None:
            lines.append(format_verdict_line(eligibility))
        if score is not None:
            lines.append(f"적합도: {score}점")
        lines.append(f"신청 URL: {doc.metadata.get('aplyUrlAddr', '정보 없음')}\n")
    knowledge = "\n".join(lines)

    logger.info("취업 정책 검색 완료 — %d건", len(policies))
    return {
        "domain_knowledge": {"employment": knowledge},
        "domain_policies": {"employment": policies},
    }
