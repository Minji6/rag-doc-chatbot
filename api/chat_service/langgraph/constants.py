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