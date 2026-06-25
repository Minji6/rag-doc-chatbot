import logging
from langchain_postgres import PGVector
from langchain.embeddings import init_embeddings
from api.common.sqlalchemy_conf import engine
from ..state import ShareState
from ..constants import (
    AGENT_CATEGORY,
    PGVECTOR_COLLECTION_NAME,
    POLICY_METADATA_FIELDS,
)

# 복지문화 도메인 전용 임계값 — 실측 거리값 기반 (공통 0.4보다 완화)
_SIMILARITY_THRESHOLD = 0.85

logger = logging.getLogger(__name__)

_CATEGORY = AGENT_CATEGORY["welfare"]

# 모듈 싱글톤 — 매 검색마다 재생성하지 않도록 import 시점에 1회만 생성.
_vectorstore = PGVector(
    embeddings=init_embeddings("openai:text-embedding-3-large"),
    collection_name=PGVECTOR_COLLECTION_NAME,
    connection=engine,
    async_mode=True,
)


def _pick_policy_fields(metadata: dict) -> dict:
    """PGVector 메타에서 화이트리스트 필드만 추려 dict 생성."""
    return {key: metadata.get(key) for key in POLICY_METADATA_FIELDS}


async def welfare_search_node(state: ShareState) -> dict:
    """복지문화 정책 검색 노드.

    PGVector에서 복지 정책을 검색해 state에 기록한다. LLM은 사용하지 않는다.
    (검색/생성 분리 구조 — 생성은 welfare_node가 담당)

    - welfare_knowledge_base: 생성 노드가 답변 근거로 읽을 정책 텍스트
    - welfare_policies: composer 후처리(비교/점수)용 raw 정책 메타
    """
    query = state["user_inquiry"]
    logger.info("복지문화 정책 검색 노드 실행 — query=%s", query[:30])

    results = await _vectorstore.asimilarity_search_with_score(
        query, k=5, filter={"category": _CATEGORY}
    )
    documents = [
        (doc, dist) for doc, dist in results if dist <= _SIMILARITY_THRESHOLD
    ]

    if not documents:
        logger.info("복지문화 정책 검색 결과 없음")
        return {
            "domain_knowledge": {"welfare": ""},
            "domain_policies": {"welfare": []},
        }

    policies = [_pick_policy_fields(doc.metadata) for doc, _ in documents]

    lines = []
    for idx, (doc, _dist) in enumerate(documents, 1):
        lines.append(f"[정책 {idx}] {doc.metadata.get('plcyNm', '')}")
        lines.append(f"내용: {doc.page_content}")
        lines.append(f"신청 URL: {doc.metadata.get('aplyUrlAddr', '정보 없음')}\n")
    knowledge = "\n".join(lines)

    logger.info("복지문화 정책 검색 완료 — %d건", len(policies))
    return {
        "domain_knowledge": {"welfare": knowledge},
        "domain_policies": {"welfare": policies},
    }
