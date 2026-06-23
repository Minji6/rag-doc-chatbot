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
    # check_eligibility 자격 검증용
    "sprtTrgtMinAge",   # 지원 대상 최소 나이
    "sprtTrgtMaxAge",   # 지원 대상 최대 나이
    "jobCd",            # 취업 상태 조건
    "srhmhldIncmCd",    # 소득 분위 조건
    "plcyAplyRgnCd",    # 신청 지역 조건
    "schoolcd",         # 학력 조건
)