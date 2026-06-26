"""정책 벡터 검색 공통 인프라 — 4개 도메인 검색 노드가 공유한다.

이전엔 housing/employment/education/welfare 검색 노드가 각자 동일한 PGVector 싱글톤과
pick_policy_fields 함수를 4벌씩 따로 들고 있었다(같은 컬렉션·임베딩·필드). 여기로 모아 1벌로 만든다.
(검색 파라미터 k/threshold·도메인 필터는 각 노드가 호출 시점에 지정하므로 동작은 동일.)
"""
from langchain_postgres import PGVector
from langchain.embeddings import init_embeddings

from api.common.sqlalchemy_conf import engine
from ..constants import PGVECTOR_COLLECTION_NAME, POLICY_METADATA_FIELDS

# 정책 벡터 검색용 PGVector 싱글톤 (import 시점 1회 생성).
# 4개 검색 노드가 동일 인스턴스를 공유 — 임베딩 객체 중복 생성 제거.
vectorstore = PGVector(
    embeddings=init_embeddings("openai:text-embedding-3-large"),
    collection_name=PGVECTOR_COLLECTION_NAME,
    connection=engine,
    async_mode=True,
)


def pick_policy_fields(metadata: dict) -> dict:
    """PGVector 메타에서 화이트리스트 필드(POLICY_METADATA_FIELDS)만 추려 dict 생성."""
    return {key: metadata.get(key) for key in POLICY_METADATA_FIELDS}
