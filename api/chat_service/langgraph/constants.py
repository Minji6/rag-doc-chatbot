# api/chat_service/langgraph/constants.py
CATEGORIES = ("일자리", "주거", "교육", "복지문화")
INQUIRY_TYPES = ("검색", "추천", "상세조회", "비교")
ROLE_USER = "user"
ROLE_GUEST = "guest"

CATEGORY_ROUTING = {
    "일자리": "employment",
    "주거": "housing",
    "교육": "education",
    "복지문화": "welfare",
}

# PGVector 정책 컬렉션명 — 도메인 에이전트 4개가 공유
PGVECTOR_COLLECTION_NAME = "youth_policy_all"

# 유사도 거리 임계값 (text-embedding-3-large 기준)
SIMILARITY_DISTANCE_THRESHOLD = 0.85

# 교육 도메인 전용 임계값 — 실측 거리값 기반 (공통 0.4보다 완화)
EDUCATION_SIMILARITY_THRESHOLD_USER = 0.7   # 로그인 유저: 관련성 있는 정책 위주
EDUCATION_SIMILARITY_THRESHOLD_GUEST = 0.9  # 게스트: 정확도보다 결과 제공 우선

# 복지문화 도메인 전용 임계값 — 실측 거리값 기반 (공통 0.4보다 완화)
WELFARE_SIMILARITY_THRESHOLD = 0.85

# 취업 도메인 전용 임계값 — 실측 거리값 기반 (공통 0.4보다 완화)
EMPLOYMENT_SIMILARITY_THRESHOLD = 0.85

# analysis_node 분류 실패/예외 시 기본 분야 (현재 라우팅은 전 분야 fan-out으로 대체되어 미사용)
DEFAULT_CATEGORY = "주거"

# inquiry_type별 검색 정책 수(k) — 4개 도메인(주거·취업·교육·복지) 공통.
# 상세조회는 특정 정책 한 개에 집중하므로 적게, 그 외(검색·추천·비교)는 기본값.
DETAIL_SEARCH_K = 2   # 상세조회: 가장 유사한 정책 위주
DEFAULT_SEARCH_K = 5  # 검색·추천·비교 기본

# 취업은 만료(마감) 정책을 걸러내므로 후보를 k배만큼 더 가져온 뒤 살아남은 것 중 k개만 사용한다.
# → 필터링에도 출력 정책 수는 다른 도메인과 동일(k)하게 유지.
EMPLOYMENT_SEARCH_OVERSAMPLE = 2


def resolve_search_k(inquiry_types: list[str] | str, requested_count: int | None = None) -> int:
    """검색 정책 수(k)를 정한다 (4개 도메인 공통).

    우선순위:
    1) 사용자가 개수를 명시했으면(requested_count 양수) 그 값을 그대로 사용 — 상한 없음.
       ("정책 3개만 추천" → policies/비교 표 모두 정확히 3개)
    2) 그 외엔 의도 기반: '상세조회'가 포함되면 좁히고(DETAIL_SEARCH_K), 아니면 기본값.

    inquiry_types는 list가 정석이나, 단일 str도 받아 하위호환을 유지한다.
    """
    if requested_count and requested_count > 0:
        return requested_count
    types = [inquiry_types] if isinstance(inquiry_types, str) else (inquiry_types or [])
    return DETAIL_SEARCH_K if "상세조회" in types else DEFAULT_SEARCH_K


# 도메인 에이전트 프롬프트 분기는 비교/상세조회 모드뿐이라 단일 모드값이면 충분하다.
# 멀티 의도(list)를 에이전트가 받을 단일 모드로 축약한다. 우선순위: 비교 > 상세조회 > 추천 > 검색.
_AGENT_MODE_PRIORITY = ("비교", "상세조회", "추천", "검색")


def agent_mode(inquiry_types: list[str] | str) -> str:
    """멀티 의도 리스트를 도메인 에이전트에 넘길 단일 모드 문자열로 축약한다."""
    types = [inquiry_types] if isinstance(inquiry_types, str) else (inquiry_types or [])
    for mode in _AGENT_MODE_PRIORITY:
        if mode in types:
            return mode
    return "검색"

# 각 도메인 에이전트가 PGVector 검색 시 filter로 사용할 category 값
# (에이전트 코드의 하드코딩을 제거하기 위해 중앙 관리)
AGENT_CATEGORY = {
    "employment": "일자리",
    "housing": "주거",
    "education": "교육",
    "welfare": "복지문화",
}

# 4개 도메인 에이전트가 공통으로 따를 출력 포맷 규칙.
# 각 에이전트 시스템 프롬프트 끝에 동일하게 주입되어 composer가 깔끔하게 조립할 수 있도록 한다.
# (도메인별 풍부한 포맷은 그대로 두고, cross-cutting 규칙만 통일)
#
# 핵심 컨트랙트: 에이전트는 "완성된 답변"이 아니라 "조립용 본문 조각(body fragment)"만 만든다.
# 인사말·분야 헤더·맺음말은 전적으로 composer 노드가 책임진다. 에이전트는 절대 출력하지 않는다.
OUTPUT_FORMAT_GUIDE = """

# 공통 출력 규칙 (반드시 준수)
- 답변은 GitHub-flavored Markdown으로 작성하세요.
- 당신의 출력은 더 큰 답변에 조립될 **본문 조각**입니다. 다음을 **절대 출력하지 마세요**:
  - 인사말: "안녕하세요", "OO님!", "OO정책팀입니다" 등 일체 금지.
  - 팀/이모지 헤더: "🏠 주거정책팀 답변:", "💼 일자리정책팀 답변:" 같은 줄 금지.
  - 분야명 헤더: "## 주거", "## 일자리" 같은 분야 헤더 금지. (시스템이 자동으로 붙입니다.)
  - 맺음말: "감사합니다", "친절하게 안내해드리겠습니다" 등 금지.
- 곧바로 첫 정책의 "### 정책명" 블록부터 시작하세요.
- 정책별 상세 헤더는 "### 정책명" 수준부터 사용하세요.
- **모든 정책마다 신청 URL을 반드시 포함하세요.** 누락 금지.
  - 정책 정보의 "신청 URL" 줄에 적힌 **실제 URL 값을 그대로** 출력하세요.
  - 형식: `신청 URL: https://example.com/apply` 처럼 raw URL 그대로 노출.
  - **절대 `[신청하기](URL)` 같은 마크다운 링크 형식으로 감싸지 마세요.**
  - 정책 정보에 URL이 없거나 "정보 없음"이면 → `신청 URL: 정보 없음`
  - URL을 임의로 만들거나 추측하지 마세요.
"""

# 비교(inquiry_type="비교") 모드에서 도메인 에이전트 user 프롬프트 끝에 주입하는 형식 재정의.
# 하이브리드 비교 설계: 정형 비교 표(금액·대상·기간·마감·URL)는 composer가 raw 메타로 직접 그리고,
# 에이전트는 "표를 그리는 일"에서 빠진다. 대신 자기 분야 전문가 관점의 짧은 정성 코멘트만 만든다.
# → 멀티에이전트(분야별 판단 유지) + 표 일관성(결정적) + 분야 교차를 동시에 달성.
COMPARISON_COMMENT_GUIDE = """

# 비교 모드 — 본문 형식 재정의 (반드시 준수)
지금은 '비교' 요청입니다. 정책들의 정형 비교 표(지원 내용·대상·신청기간·마감·신청 URL)는
시스템이 데이터로 직접 그립니다. 따라서 다음을 **절대 하지 마세요**:
- 마크다운 표(`| ... |`)를 그리지 마세요.
- "### 정책명" 블록을 만들지 마세요.
대신 **당신 분야 전문가 관점에서** 이 정책들을 비교·선택할 때의 핵심 판단 포인트를
**1~2문장**으로만 작성하세요. (예: 어떤 상황이면 어느 정책이 유리한지, 놓치기 쉬운 자격·마감 주의점)
인사말·맺음말·헤더 없이 곧바로 코멘트 문장만 출력하세요.
"""

# 상세조회(inquiry_type="상세조회") 모드에서 도메인 에이전트 user 프롬프트 끝에 주입.
# 사용자가 특정 정책을 콕 집어 물었을 때(또는 후속질문으로 한 정책이 특정됐을 때),
# 그 정책 하나를 깊이 있게 안내하도록 유도한다. (검색/추천의 나열형과 구분)
OUTPUT_DETAIL_GUIDE = """

# 상세조회 모드 — 깊이 있는 단일 정책 안내 (반드시 준수)
지금은 '상세조회' 요청입니다. 사용자가 특정 정책을 콕 집어 물었을 가능성이 높습니다.
- [정책 정보]에서 질문에 가장 부합하는 **그 정책 1개에 집중**해 깊이 있게 설명하세요.
  여러 개가 검색됐어도 핵심 1개를 우선 안내하고, 유사 정책은 맨 끝에 한 줄로만 덧붙이세요.
- 다음 항목을 가능한 한 빠짐없이 포함하세요(정보가 없으면 "정보 없음"으로 표기, 지어내지 말 것):
  정책 목적/배경, 지원 내용(금액·기간·방식 등 구체 수치), 참여 자격(연령·소득·지역·기타 조건),
  제출 서류, 신청 절차(단계별), 신청 기간과 마감일(D-day), 문의처, 신청 시 유의사항.
"""

# 추천(inquiry_type="추천") 모드에서 도메인 에이전트 user 프롬프트 끝에 주입 (4개 도메인 공용).
# 검색 노드가 벡터 유사도를 0~100 적합도(관련도)로 환산해 [정책 정보]에 "적합도: N점" 줄로 넣어둔다.
# 에이전트는 각 정책 블록에 (1) 추천 이유 (2) 적합도 두 줄을 붙인다.
# '추천 이유'는 반드시 긍정 서술로만 — 자격 판정("~라서 안 맞음")은 금지(적합도 점수가 그 역할).
RECOMMEND_GUIDE = """

# 추천 모드 — 각 정책 블록에 아래 두 줄 반드시 포함 (반드시 준수)
- 추천 이유: 이 정책의 강점·혜택과 어떤 사람에게 유용한지 한 줄로, 긍정적으로 서술하세요.
  사용자가 자격 조건(지역·나이·소득·학력 등)에 맞는지 여부는 이 칸에서 판단하지 마세요.
  ("~라서 맞지 않습니다/추천하지 않습니다/충족되지 않습니다" 류 표현 금지 — 자격 판단은 적합도 점수가 보여줍니다.)
- 적합도: [정책 정보]에 "적합도: N점" 줄이 있으면 그 값을 "N점"으로 그대로 표시하세요.
  점수를 지어내거나 다른 값으로 바꾸지 말고, "적합도: N점" 줄이 없는 정책은 이 항목을 생략하세요.
"""

# 도메인 에이전트가 *_result["policies"]에 담을 PGVector 메타 필드 화이트리스트.
# compare_policies / policy_priority_score 등 후속 tool이 노이즈 없이 받을 수 있도록 최소 필드만 추림.
# 필요한 필드 누락 시 여기에 추가.
POLICY_METADATA_FIELDS = (
    "plcyNo",        # 정책번호 (식별자)
    "plcyNm",        # 정책명
    "category",      # 분야
    "sub_category",  # 세부분야
    "plcyExplnCn",   # 정책 설명
    "plcySprtCn",    # 지원 내용
    "ptcpPrpTrgtCn", # 참여 대상
    "addAplyQlfcCndCn", # 추가 신청 자격 조건
    "aplyUrlAddr",   # 신청 URL
    "aplyPrdSeCd",   # 신청기간 구분 (특정기간/상시/마감)
    "aplyYmd",       # 신청 기간
    "bizPrdBgngYmd", # 사업 시작일
    "bizPrdEndYmd",  # 사업 종료일
    # 주거 정책 자격 진단용 필드 (housing_search_node에서 사용)
    "sprtTrgtMinAge",    # 최소 나이
    "sprtTrgtMaxAge",    # 최대 나이
    "sprtTrgtAgeLmtYn",  # 연령 제한 여부 (Y/N)
    "zipCd",             # 지역 코드
    "mrgSttsCd",         # 혼인 상태 코드
    # check_eligibility 자격 검증용
    "jobCd",            # 취업 상태 조건
    "srhmhldIncmCd",    # (구) 소득 조건 — 미사용, 하위호환 위해 유지
    "earnCndSeCd",      # 소득 조건 구분 (무관/연소득/기타)
    "earnMinAmt",       # 소득 최소 금액 (만원, 연소득일 때)
    "earnMaxAmt",       # 소득 최대 금액 (만원, 연소득일 때)
    "plcyAplyRgnCd",    # 신청 지역 조건
    "schoolcd",         # 학력 조건
    # 카드 UI 표시용
    "aplyMthdCn",       # 신청 방법
    "cnsgNmCn",         # 문의처
)