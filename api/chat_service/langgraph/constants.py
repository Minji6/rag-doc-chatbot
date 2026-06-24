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

# gather_node에서 category → 어느 *_result 필드를 읽을지 매핑
CATEGORY_RESULT_FIELD = {
    "일자리": "employment_result",
    "주거": "housing_result",
    "교육": "education_result",
    "복지문화": "welfare_result",
}

# PGVector 정책 컬렉션명 — 도메인 에이전트 4개가 공유
PGVECTOR_COLLECTION_NAME = "youth_policy_all"

# 유사도 거리 임계값 (text-embedding-3-large 기준, 지침서 §22-6)
SIMILARITY_DISTANCE_THRESHOLD = 0.4

# 교육 도메인 전용 임계값 — 실측 거리값 기반 (공통 0.4보다 완화)
EDUCATION_SIMILARITY_THRESHOLD_USER = 0.7   # 로그인 유저: 관련성 있는 정책 위주
EDUCATION_SIMILARITY_THRESHOLD_GUEST = 0.9  # 게스트: 정확도보다 결과 제공 우선

# analysis_node 분류 실패/예외 시 기본 분야 — 라우팅 함수 fallback에 사용
DEFAULT_CATEGORY = "주거"

# 각 도메인 에이전트가 PGVector 검색 시 filter로 사용할 category 값
# (에이전트 코드의 하드코딩을 제거하기 위해 중앙 관리)
AGENT_CATEGORY = {
    "employment": "일자리",
    "housing": "주거",
    "education": "교육",
    "welfare": "복지문화",
}

# 4개 도메인 에이전트가 공통으로 따를 출력 포맷 규칙.
# 각 에이전트 시스템 프롬프트 끝에 동일하게 주입되어 gather가 깔끔하게 합칠 수 있도록 한다.
# (도메인별 풍부한 포맷은 그대로 두고, cross-cutting 규칙만 통일)
OUTPUT_FORMAT_GUIDE = """

# 공통 출력 규칙 (반드시 준수)
- 답변은 GitHub-flavored Markdown으로 작성하세요.
- 답변 최상단에 "## 주거", "## 일자리" 같은 분야명 헤더를 **직접 작성하지 마세요**.
  (분야 헤더는 시스템이 자동으로 붙입니다.)
- 정책별 상세 헤더는 "### 정책명" 수준부터 사용하세요.
- **모든 정책마다 신청 URL을 반드시 포함하세요.** 누락 금지.
  - 정책 정보의 "신청 URL" 줄에 적힌 **실제 URL 값을 그대로** 출력하세요.
  - 형식: `신청 URL: https://example.com/apply` 처럼 raw URL 그대로 노출.
  - **절대 `[신청하기](URL)` 같은 마크다운 링크 형식으로 감싸지 마세요.**
  - 정책 정보에 URL이 없거나 "정보 없음"이면 → `신청 URL: 정보 없음`
  - URL을 임의로 만들거나 추측하지 마세요.
- 답변 도입부 인사말은 1~2줄로 간결히.
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
    "srhmhldIncmCd",    # 소득 분위 조건
    "plcyAplyRgnCd",    # 신청 지역 조건
    "schoolcd",         # 학력 조건
)