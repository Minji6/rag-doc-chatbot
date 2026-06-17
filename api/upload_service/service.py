import asyncio
import logging
import selectors
from pathlib import Path
from typing import Annotated

from fastapi import Depends
from langchain_postgres import PGVector
from langchain.embeddings import init_embeddings

class EmbeddingService:
    """
    유사도 검색 테스트 서비스
    """

    # 1. 초기화 메서드
    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.EmbeddingService")
        self.embeddings = init_embeddings(model="openai:text-embedding-3-large")

    # ------------------------------------------------------------------
    async def similarity_search(
        self,
        collection_name: str,
        query: str,
        k: int = 3,
    ) -> str:
        """저장된 벡터에서 유사도 검색을 수행한다(연동 테스트용)."""
        from api.common.sqlalchemy_conf import engine

        vectorstore = PGVector(
            embeddings=self.embeddings,
            collection_name=collection_name,
            connection=engine,
            async_mode=True,
        )
        results = await vectorstore.asimilarity_search_with_score(query, k)

        if not results:
            return "관련된 문서를 찾을 수 없습니다."
        return "\n".join(
            f"거리: {distance}, 내용: {doc.page_content[:30]}" for doc, distance in results
        )
    
    
    ##################################################
    # 🔥 추가: welfare 전용 wrapper
    ##################################################
    async def search_welfare_policy(
        self,
        query: str,
        category: str,
        collection_name: str,
        k: int,
        distance_threshold: float,
        user_role: str = "guest",
        user_profile=None,
    ) -> str:

        final_query = query

        # ✔ user profile 반영
        if user_role == "user" and user_profile is not None:
            parts = []

            if user_profile.age is not None:
                parts.append(f"{user_profile.age}세")

            if user_profile.gender:
                parts.append(user_profile.gender)

            if user_profile.is_university_graduate is not None:
                parts.append(
                    "대학 졸업자" if user_profile.is_university_graduate else "대학 미졸업자"
                )

            if user_profile.major:
                parts.append(f"{user_profile.major} 전공")

            if user_profile.region:
                parts.append(user_profile.region)

            if user_profile.income_level is not None:
                parts.append(f"기준 중위소득 {user_profile.income_level}%")

            if parts:
                final_query = f"{query} ({', '.join(parts)})"

        return await self.similarity_search(
            collection_name=collection_name,
            query=final_query,
            k=k,
        )


# 의존성 타입 별칭
EmbeddingServiceDep = Annotated[EmbeddingService, Depends(EmbeddingService)]


# 최상위 모듈로 직접 실행하는 경우 (연동 테스트용)
if __name__ == "__main__":
    import sys

    sys.path.append(str(Path(__file__).parents[2]))

    logging.basicConfig(level=logging.INFO)
    service = EmbeddingService()

    # Windows는 기본적으로 ProactorEventLoop를 사용
    # - psycopg 비동기 드라이버가 이를 지원하지 않아 오류가 발생
    # - loop_factory로 SelectorEventLoop를 지정해서 호환성 문제를 해결.
    def _selector_loop() -> asyncio.AbstractEventLoop:
        return asyncio.SelectorEventLoop(selectors.SelectSelector())

    # 2) 검색 테스트
    result = asyncio.run(
        service.similarity_search(
            collection_name="youth_policy_welfare_culture",
            query="저소득 청년을 위한 금융 지원 정책이 있나요?",
            k=3,
        ),
        loop_factory=_selector_loop,
    )
    print(result)