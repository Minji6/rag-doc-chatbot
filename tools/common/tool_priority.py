import logging
import re
from langchain.tools import tool

from tools.common.tool_rag_search import search_policy_impl, search_web_supplement_impl
from api.chat_service.langgraph.constants import POLICY_METADATA_FIELDS

logger = logging.getLogger(__name__)


def _extract_policy_names(rag_result: str) -> list[str]:
    """
    RAG 결과에서 정책명 목록을 추출합니다.
    "[N] 정책명" 형식에서 정책명만 추출.
    """
    policy_names = []
    for line in rag_result.split("\n"):
        match = re.match(r"\[\d+\]\s+(.+)", line)
        if match:
            policy_names.append(match.group(1))
    return policy_names


def _extract_rag_count(rag_result: str) -> int:
    """
    RAG 결과에서 "총 N건 검색됨"에서 N을 추출합니다.
    """
    match = re.search(r"총\s+(\d+)건\s+검색됨", rag_result)
    if match:
        return int(match.group(1))
    return 0


@tool
async def answer_with_priority(query: str, category: str) -> str:
    """
    RAG 1순위 → 웹 보충 → 최종 답변을 생성합니다.

    Step 1. search_policy로 RAG 검색 (1순위)
    Step 2. [RAG_FALLBACK] 신호 확인
            - 신호 없음: RAG만 사용
            - 신호 있음: search_web_supplement로 부족분 채우기
    Step 3. 웹 결과를 "[LLM]\n..." 형식으로 포장

    Args:
        query: 사용자 질문
        category: 정책 분야 ("복지문화" / "일자리" / "주거" / "교육")

    Returns:
        str: "[RAG]\n정책들" 또는 "[RAG]\n정책들\n\n[LLM]\n웹정책들"
    """
    try:
        logger.info(f"answer_with_priority 호출: query='{query}', category='{category}'")

        # Step 1: RAG 검색 (1순위)
        rag_result = await search_policy_impl(query, category)

        # Step 2: [RAG_FALLBACK] 신호 확인
        if "[RAG_FALLBACK]" not in rag_result:
            # RAG 성공, 부족분 없음
            logger.info(f"RAG 충분 - category={category}")
            return f"[RAG]\n{rag_result}"

        # RAG 부족 → 웹 보충 필요
        logger.info(f"RAG 부족 - 웹 보충 시작 - category={category}")

        # RAG 결과에서 정책명과 개수 추출
        rag_lines = rag_result.replace("[RAG_FALLBACK]\n", "").strip()
        rag_policy_names = _extract_policy_names(rag_lines)
        rag_count = _extract_rag_count(rag_lines)
        web_needed_count = 5 - rag_count

        # Step 3: search_web_supplement_impl로 부족분 채우기
        web_result = await search_web_supplement_impl(
            query=query,
            exclude_titles=rag_policy_names,
            count=web_needed_count
        )

        # Step 4: 결과 조합 (웹 결과를 "[LLM]\n..." 형식으로 포장)
        # Note: RAG/WEB 결과는 tool_rag_search에서 이미 포맷됨 (POLICY_METADATA_FIELDS 준수)
        combined_result = f"[RAG]\n{rag_lines}"

        if "웹 검색 결과가 없습니다" not in web_result and "[오류]" not in web_result:
            combined_result += f"\n\n[LLM]\n{web_result}"
            logger.info(f"RAG+웹 검색 완료: {rag_count}개(RAG) + {web_needed_count}개(WEB)")
        else:
            logger.info(f"웹 보충 결과 없음: {web_result}")

        return combined_result

    except Exception as e:
        logger.error(f"answer_with_priority 실행 중 오류: {str(e)}")
        return f"[오류] 답변 생성 중 문제가 발생했습니다: {str(e)}"
