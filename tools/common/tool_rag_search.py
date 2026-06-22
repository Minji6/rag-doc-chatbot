import logging
from contextvars import ContextVar
from langchain.tools import tool
from langchain_postgres import PGVector
from langchain.embeddings import init_embeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
from api.common.sqlalchemy_conf import engine
from api.chat_service.langgraph.constants import (
    PGVECTOR_COLLECTION_NAME,
    SIMILARITY_DISTANCE_THRESHOLD,
    POLICY_METADATA_FIELDS,
)

logger = logging.getLogger(__name__)

_last_rag_policies: ContextVar[list[dict]] = ContextVar("last_rag_policies", default=[])


def _pick_policy_fields(metadata: dict) -> dict:
    return {key: metadata.get(key) for key in POLICY_METADATA_FIELDS}


async def search_policy_impl(
    query: str, category: str, k: int = 5
) -> tuple[str, list[dict]]:
    """비동기 PGVector 검색 구현 (tool_priority.py에서 직접 호출용).

    Returns:
        tuple: (포맷된 텍스트, POLICY_METADATA_FIELDS 준수 structured metadata)
    """
    vectorstore = PGVector(
        embeddings=init_embeddings("openai:text-embedding-3-large"),
        collection_name=PGVECTOR_COLLECTION_NAME,
        connection=engine,
        async_mode=True,
    )

    results = await vectorstore.asimilarity_search_with_score(
        query,
        k=k,
        filter={"category": category},
    )

    filtered = [
        (doc, score)
        for doc, score in results
        if score <= SIMILARITY_DISTANCE_THRESHOLD
    ]

    # 구조화된 metadata 추출 (POLICY_METADATA_FIELDS 준수)
    filtered_metadata = [_pick_policy_fields(doc.metadata) for doc, _ in filtered]
    _last_rag_policies.set(filtered_metadata)

    # 결과 포맷팅
    if not filtered:
        return "[RAG_FALLBACK]", []

    lines: list[str] = []
    for i, (doc, _) in enumerate(filtered, start=1):
        m = doc.metadata
        lines.append(
            f"[{i}] {m.get('plcyNm', '')}\n"
            f"설명: {m.get('plcyExplnCn', '')}\n"
            f"지원내용: {m.get('plcySprtCn', '')}\n"
            f"신청방법: {m.get('plcyAplyMthdCn', '')}"
        )

    lines.append(f"\n총 {len(filtered)}건 검색됨")
    result_text = "\n\n".join(lines)

    # 결과가 k개 미만이면 [RAG_FALLBACK] 신호 추가
    if len(filtered) < k:
        result_text = f"[RAG_FALLBACK]\n{result_text}"
        logger.info(f"RAG 검색 완료 (부족): {category}, {len(filtered)}건 (요청: {k}건)")
    else:
        logger.info(f"RAG 검색 완료: {category}, {len(filtered)}건")

    return result_text, filtered_metadata


@tool
async def search_policy(query: str, category: str, k: int = 5) -> str:
    """
    PGVector에서 청년 정책을 검색합니다. (RAG 1순위, Agent용)
    내부적으로 search_policy_impl을 호출합니다.

    Args:
        query: 검색할 질문이나 키워드
        category: 정책 분야 ("복지문화" / "일자리" / "주거" / "교육")
        k: 반환할 최대 정책 수 (기본값 5)

    Returns:
        str: 검색된 정책 목록 텍스트. 결과 없으면 "[RAG_FALLBACK]" 신호 문자열
    """
    try:
        return await search_policy_impl(query, category, k)
    except Exception as e:
        logger.error(f"RAG 검색 실패: {e}")
        return f"[오류] RAG 검색 중 문제 발생: {str(e)}"


async def search_web_supplement_impl(
    query: str,
    exclude_titles: list[str],
    count: int,
) -> str:
    """
    웹 검색으로 정책을 보완하는 구현 (tool_priority.py에서 직접 호출용)
    """
    try:
        tavily_search = TavilySearchResults(
            max_results=count * 2,
            search_depth="basic",
        )

        results = await tavily_search.ainvoke({"query": query})

        exclude_set = set(exclude_titles)
        collected: list[str] = []

        for result in results:
            if len(collected) >= count:
                break
            title = result.get("title", "")
            if title in exclude_set:
                continue
            collected.append(
                f"[{len(collected) + 1}] {title}\n"
                f"출처: {result.get('url', '')}\n"
                f"요약: {result.get('content', '')}"
            )

        if not collected:
            return "웹 검색 결과가 없습니다."

        result_text = "\n\n".join(collected)
        logger.info(f"웹 보충 검색 완료: {len(collected)}건")
        return result_text

    except Exception as e:
        logger.error(f"웹 검색 실패: {e}")
        return f"[오류] 웹 검색 중 문제 발생: {str(e)}"


@tool
async def search_web_supplement(
    query: str,
    exclude_titles: list[str],
    count: int,
) -> str:
    """
    RAG 검색 결과가 부족할 때 웹 검색으로 정책을 보완합니다. (Agent용)
    내부적으로 search_web_supplement_impl을 호출합니다.

    Args:
        query: 검색할 질문이나 키워드
        exclude_titles: 이미 검색된 정책명 목록 (중복 제외용)
        count: 추가로 가져올 정책 수

    Returns:
        str: 추가 정책 목록 텍스트
    """
    return await search_web_supplement_impl(query, exclude_titles, count)