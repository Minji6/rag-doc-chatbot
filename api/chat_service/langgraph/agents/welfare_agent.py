import logging
from typing import Annotated

from fastapi import Depends
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_postgres import PGVector
from langchain.embeddings import init_embeddings

from api.common.sqlalchemy_conf import engine
from api.chat_service.langgraph.state import ShareState

# 로거 생성
logger = logging.getLogger(__name__)


##############################################################
# 도구 정의
##############################################################
@tool
async def search_policy(query: str, runtime: ToolRuntime) -> str:
    """
    복지(금융·문화·예술) 분야 청년 정책을 PGVector에서 검색합니다.

    user_role이 "user"이고 state에 user_profile이 있으면, 나이/성별/
    대학졸업유무/전공 등 프로필 정보를 검색 쿼리에 반영해 맞춤 검색을
    수행합니다. user_role이 "guest"이거나 user_profile이 없으면
    질문 내용만으로 기본 검색을 수행합니다.

    Args:
        query: 검색할 질문이나 키워드
    Returns:
        str: 검색된 정책 내용
    """
    user_role = runtime.state.get("user_role", "guest")
    user_profile = runtime.state.get("user_profile")

    search_query = query

    # user이고 프로필이 있는 경우에만 쿼리에 프로필 조건을 반영 (맞춤 검색)
    if user_role == "user" and user_profile is not None:
        profile_parts = []
        if user_profile.age is not None:
            profile_parts.append(f"{user_profile.age}세")
        if user_profile.gender:
            profile_parts.append(user_profile.gender)
        if user_profile.is_university_graduate is not None:
            profile_parts.append(
                "대학 졸업자" if user_profile.is_university_graduate else "대학 미졸업자"
            )
        if user_profile.major:
            profile_parts.append(f"{user_profile.major} 전공")
        if user_profile.region:
            profile_parts.append(user_profile.region)
        if user_profile.income_level:
            profile_parts.append(user_profile.income_level)

        if profile_parts:
            search_query = f"{query} ({', '.join(profile_parts)})"

        logger.info(f"맞춤 검색(user) 쿼리: {search_query}")
    else:
        logger.info(f"기본 검색(guest) 쿼리: {search_query}")

    vectorstore = PGVector(
        # 텍스트 -> 숫자 변환을 위한 모델 임베딩
        embeddings=init_embeddings(model="openai:text-embedding-3-large"),
        collection_name="울산_청년정책",  # 추후 컬렉션 내용 변경
        connection=engine,
        async_mode=True,
    )

    # 카테고리가 '복지'인 항목중에서 유사도 기반 검색
    results = await vectorstore.asimilarity_search_with_score(
        search_query,
        k=5,
        filter={"category": "복지"},
    )

    # 임시 임계값 (데이터 교체 후 변경 가능)
    documents = [(doc, dist) for doc, dist in results if dist < 0.65]

    if not documents:
        return f"'{search_query}' 관련 복지 정책을 찾을 수 없습니다."

    lines = []
    for idx, (doc, dist) in enumerate(documents, 1):
        lines.append(f"[정책: {idx}] {doc.metadata.get('title', '')}")
        lines.append(f"내용: {doc.page_content}\n")

    return "\n".join(lines)


##############################################################
# Agent 클래스 정의
##############################################################
class WelfareAgent:
    """
    복지(금융·문화·예술) 분야 청년정책 도메인 에이전트.

    슈퍼바이저(analysis_node)가 category="복지"로 분류한 요청을 이 에이전트로
    라우팅하면, search_policy 도구가 state의 user_role/user_profile을
    직접 읽어(ToolRuntime) guest/user에 따른 검색 깊이를 자동으로 조절한다.
    """

    def __init__(self, model: str = "openai:gpt-4o-mini") -> None:
        self.logger = logging.getLogger(f"{__name__}.WelfareAgent")
        self.agent = create_agent(
            model=model,
            tools=[search_policy],
            state_schema=ShareState,  # type: ignore
            system_prompt="당신은 청년 복지(금융·문화·예술) 정책 전문가입니다. 복지 관련 정책만 안내하세요.",
        )

    async def run(
        self,
        question: str,
        user_role: str = "guest",
        user_profile=None,
    ) -> str:
        """복지 에이전트를 실행한다.

        Args:
            question: 사용자 질문
            user_role: "guest"(기본값) 또는 "user". search_policy 도구가
                runtime.state를 통해 이 값을 직접 읽어 검색 방식을 결정한다.
            user_profile: user_role이 "user"일 때 state에서 가져온 UserProfile.
                guest이거나 정보가 없으면 None.

        Returns:
            str: 에이전트의 최종 응답
        """
        result = await self.agent.ainvoke(
            {
                "messages": [{"role": "user", "content": question}],
                "user_role": user_role,
                "user_profile": user_profile,
            }  # type: ignore
        )
        return result["messages"][-1].content


# 의존성 주입을 위한 타입 힌트 정의
WelfareAgentDep = Annotated[WelfareAgent, Depends(WelfareAgent)]