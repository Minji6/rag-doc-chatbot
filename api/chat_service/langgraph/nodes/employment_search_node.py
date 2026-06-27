import logging
from datetime import date
from ..state import ShareState
from ..tools.dday import end_date_from
from ..tools.policy_search import vectorstore as _vectorstore, pick_policy_fields as _pick_policy_fields
from ..tools.eligibility import check_policy_eligibility, format_verdict_line, eligibility_sort_key
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
    # 출력 정책 수(k)는 4개 도메인 공통(resolve_search_k). 단 취업은 만료 정책을 걸러내므로
    # 후보를 k배(oversample)만큼 더 가져온 뒤, 살아남은 것 중 k개만 사용한다.
    k = resolve_search_k(state.get("inquiry_type", "검색"))
    fetch_k = k * EMPLOYMENT_SEARCH_OVERSAMPLE
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

    # 자격 진단 기준으로 정렬: eligible → unknown → ineligible (게스트는 검색 순서 유지)
    documents_with_dday.sort(key=lambda item: eligibility_sort_key(item[0].metadata, user_profile))

    documents_with_dday = documents_with_dday[:k]  # 출력은 공통 k개로 제한

    if not documents_with_dday:
        logger.info("취업 정책 검색 결과 없음")
        return {
            "domain_knowledge": {"employment": ""},
            "domain_policies": {"employment": []},
        }

    policies = [_pick_policy_fields(doc.metadata) for doc, _, _ in documents_with_dday]

    lines = []
    for idx, (doc, _dist, dday) in enumerate(documents_with_dday, 1):
        lines.append(f"[정책 {idx}] {doc.metadata.get('plcyNm', '')}")
        lines.append(f"내용: {doc.page_content}")
        if dday:
            lines.append(f"신청기간: {dday}")

        # 자격 진단 라벨 (로그인 유저만, 게스트는 None 반환되어 스킵)
        eligibility = check_policy_eligibility(doc.metadata, user_profile)
        if eligibility is not None:
            lines.append(format_verdict_line(eligibility))

        lines.append(f"신청 URL: {doc.metadata.get('aplyUrlAddr', '정보 없음')}\n")
    knowledge = "\n".join(lines)

    logger.info("취업 정책 검색 완료 — %d건", len(policies))
    return {
        "domain_knowledge": {"employment": knowledge},
        "domain_policies": {"employment": policies},
    }
