import logging
from datetime import date
from ..state import ShareState
from ..tools.dday import end_date_from
from ..tools.policy_search import vectorstore as _vectorstore, pick_policy_fields as _pick_policy_fields
from ..constants import AGENT_CATEGORY, resolve_search_k

# 취업 도메인 전용 임계값 — 실측 거리값 기반 (공통 0.4보다 완화)
_SIMILARITY_THRESHOLD = 0.85

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
    # 취업은 만료 정책을 걸러내므로 recall 확보용으로 기본 k를 넉넉히(10) 가져온다.
    # 상세조회만 좁혀(resolve_search_k) 특정 정책에 집중.
    k = resolve_search_k(state.get("inquiry_type", "검색"), default=10)
    logger.info("취업 정책 검색 노드 실행 — k=%d, query=%s", k, query[:30])

    results = await _vectorstore.asimilarity_search_with_score(
        query, k=k, filter={"category": _CATEGORY}
    )

    documents_with_dday: list[tuple] = []
    for doc, dist in results:
        if dist > _SIMILARITY_THRESHOLD:
            continue
        label = _dday_label(
            doc.metadata.get("aplyPrdSeCd", ""),
            doc.metadata.get("aplyYmd", ""),
        )
        if label is None:  # 마감 / 기간 만료
            continue
        documents_with_dday.append((doc, dist, label))

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
        lines.append(f"신청 URL: {doc.metadata.get('aplyUrlAddr', '정보 없음')}\n")
    knowledge = "\n".join(lines)

    logger.info("취업 정책 검색 완료 — %d건", len(policies))
    return {
        "domain_knowledge": {"employment": knowledge},
        "domain_policies": {"employment": policies},
    }
