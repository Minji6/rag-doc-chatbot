from datetime import date
import logging
from typing import Literal, TypedDict
from langchain.embeddings import init_embeddings
from langchain_postgres import PGVector
from api.common.sqlalchemy_conf import engine
from ..state import ShareState
from ..constants import AGENT_CATEGORY, PGVECTOR_COLLECTION_NAME, POLICY_METADATA_FIELDS, SIMILARITY_DISTANCE_THRESHOLD

# 로거 생성
logger = logging.getLogger(__name__)

# 카테고리 [주거]로 분류
_CATEGORY = AGENT_CATEGORY["housing"]

# 모듈 싱글톤 - import 시점에 1회만 생성
_vectorstore = PGVector(
    embeddings=init_embeddings("openai:text-embedding-3-large"),
    collection_name=PGVECTOR_COLLECTION_NAME,
    connection=engine,
    async_mode=True
)

##############################################################
# 주거 자격 사전 진단
#   - LLM 미사용, 정규식 + 키워드 기반
#   - user_profile이 비어있으면(=게스트) 진단 스킵
#   - eliginble (신청가능) / ineliginle (신청불가) / unknown (판단불가) 로 분류 
##############################################################
Verdict = Literal["eligible", "ineligible", "unknown"]

class EligibilityResult(TypedDict):
    verdict: Verdict
    reasons: list[str]    

# 사용자의 만나이를 계산하는 함수
def _calc_age(birth_date: str | None) -> int | None:
    if not birth_date:
        return None
    try:
        y, m, d = map(int, birth_date[:10].split("-"))
        born = date(y, m, d)
    except (ValueError, AttributeError):
        return None
    
    # 오늘 날짜 기준으로 만나이 계산
    today = date.today()
    age = today.year - born.year
    # 튜플 계산을 통해 월, 일을 순서대로 비교 (만약 생일이 지나지 않았으면 나이 -1)
    if (today.month, today.day) < (born.month, born.day):
        age -= 1
    
    return age

# 나이제한 검사
def _check_age(meta: dict, age: int | None) -> tuple[Verdict, str] :
    # 정책이 연령 제한 없음을 명시하면 무조건 통과
    if meta.get("sprtTrgtAgeLmtYn") == "N":
        return "eligible", "연령 ✓ (제한 없음)"
    
    if age is None :
        return "unknown", "연령: 생년월일 정보 없음"
    
    lo_raw, hi_raw = meta.get("sprtTrgtMinAge"), meta.get("sprtTrgtMaxAge")
    lo = int(lo_raw) if lo_raw else None
    hi = int(hi_raw) if hi_raw else None
    
    # 데이터에 연령 정보가 아예 없을 경우 -> unknown 
    if lo is None and hi is None :
        return "unknown", "연령조건 : 정책 데이터 부족"
    
    # 있는 조건만 검사 → 하나라도 위배되면 미충족
    if lo is not None and age < lo:
        return "ineligible", f"연령 ✗ (만 {age}세, 최소 {lo}세 이상)"
    if hi is not None and age > hi:
        return "ineligible", f"연령 ✗ (만 {age}세, 최대 {hi}세 이하)"
    
    # 통과 — 메시지에 알려진 조건만 표시
    bound_msg = f"{lo or '?'}~{hi or '?'}세"
    return "eligible", f"연령 ✓ (만 {age}세, 요건 {bound_msg})"

# 지역 검사
def _check_region(meta: dict, user_region: str|None) -> tuple[Verdict, str]:
    # 지역 코드 읽어오기
    region_code = meta.get("zipCd") or ""
    
    # "전국"인 경우 -> eligible
    if "전국" in region_code:
        return "eligible", "지역 ✓ (전국)"
    # 데이터가 빈 값 ->  unknown
    if not region_code:
        return "unknown", "지역 요건: 정책 데이터 부족"
    # 사용자가 빈 값 -> unknwon
    if not user_region:
        return "unknown", "지역: 사용자 거주지 정보 없음"
    
    # 양방향 substring — "전북" / "전라북도" / "전북특별자치도" 어떤 형태든 매칭되게
    parts = [p.strip() for p in region_code.split(",") if p.strip()]
    # 지역 조건에 맞는 경우
    for p in parts:
        if p in user_region or user_region in p:
            return "eligible", f"지역 ✓ ({user_region})"
    # 지역 조건에 맞지 않는 경우
    return "ineligible", f"지역 ✗ (요건: {region_code} / 거주지: {user_region})"

# 결혼여부 검사
def _check_marriage(meta: dict, user_mrg: str | None) -> tuple[Verdict, str]:
    pol_mrg = meta.get("mrgSttsCd")
    # 정책이 혼인 무관 -> eligible
    if pol_mrg == "제한없음":
        return "eligible", "혼인 ✓ (제한없음)"
    # 데이터가 빈값 -> unknwon
    if not pol_mrg:
        return "unknown", "혼인 요건: 정책 데이터 부족"
    # 사용자가 혼인여부 데이터를 채우지 않은 경우
    if not user_mrg:
        return "unknown", "혼인: 사용자 정보 없음"
    # 사용자의 혼인여부와 정책의 혼인여부가 일치하는 경우
    if user_mrg == pol_mrg:
        return "eligible", f"혼인 ✓ ({user_mrg})"
    # 사용자의 혼인여부가 정책의 혼인제한에 걸리는 경우 
    return "ineligible", f"혼인 ✗ (요건: {pol_mrg} / 현재: {user_mrg})"


# 정책 자격 진단
def _check_policy_eligibility(
    metadata: dict, user_profile: dict | None
) -> EligibilityResult | None:
    """단일 정책 자격 진단. user_profile이 비어있으면 None."""
    if not user_profile:
        return None
    
    # 나이 계산 후 검사
    age = _calc_age(user_profile.get("birth_date"))
    checks: list[tuple[Verdict, str]] = [
        _check_age(metadata, age),
        _check_region(metadata, user_profile.get("zipcd")),
        _check_marriage(metadata, user_profile.get("mrgsttscd")),
    ]

    reasons = [r for _, r in checks]
    verdicts = {v for v, _ in checks}
    if "ineligible" in verdicts:
        final: Verdict = "ineligible"
    elif "unknown" in verdicts:
        final = "unknown"
    else:
        final = "eligible"
    return EligibilityResult(verdict=final, reasons=reasons)

# 자격진단 결과 출력 형식
def _format_verdict_line(result: EligibilityResult) -> str:
    icon = {
        "eligible":   "✅ 자격 적합",
        "ineligible": "❌ 자격 미충족",
        "unknown":    "❓ 자격 확인 필요",
    }[result["verdict"]]
    return f"{icon} — {' | '.join(result['reasons'])}"

def _pick_policy_fields(metadata: dict) -> dict:
    """
    PGVector 메타에서 화이트리스트 필드만 추려 dict 생성
    """
    return {key: metadata.get(key) for key in POLICY_METADATA_FIELDS}

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
    logger.info("주거 정책 검색 노드 실행 - query=%s", query[:30])
    
    results = await _vectorstore.asimilarity_search_with_score(
        query, k=5, filter={"category": _CATEGORY}
    )
    
    documents = [
        (doc, dist) for doc, dist in results if dist <= SIMILARITY_DISTANCE_THRESHOLD
    ]
    
    if not documents:
        logger.info("주거 정책 검색 결과 없음")
        return {"domain_knowledge": {"housing": ""}, "domain_policies": {"housing": []}}
    
    # 자격 진단 결과 기준으로 정렬: eligible → unknown → ineligible
    def _sort_key(item):
        doc, _dist = item
        e = _check_policy_eligibility(doc.metadata, user_profile)
        if e is None:
            return 2  # 게스트면 검색 순서 유지
        return {"eligible": 0, "unknown": 1, "ineligible": 2}[e["verdict"]]

    documents.sort(key=_sort_key)
    
    policies = [_pick_policy_fields(doc.metadata) for doc, _ in documents]
    
    lines = []
    for idx, (doc, _dist) in enumerate(documents, 1):
        lines.append(f"[정책 {idx}] {doc.metadata.get('plcyNm', '')}")
        lines.append(f"내용: {doc.page_content}")
        
        # 자격 진단 라벨 (로그인 유저만, 게스트는 None 반환되어 스킵)
        eligibility = _check_policy_eligibility(doc.metadata, user_profile)
        if eligibility is not None:
            lines.append(_format_verdict_line(eligibility))
            
        lines.append(f"신청 URL: {doc.metadata.get('aplyUrlAddr', '정보 없음')}\n")
    knowledge = "\n".join(lines)
    
    logger.info("주거 정책 검색 완료 - %d건", len(policies))
    return {
        "domain_knowledge": {"housing": knowledge},
        "domain_policies": {"housing": policies},
    }