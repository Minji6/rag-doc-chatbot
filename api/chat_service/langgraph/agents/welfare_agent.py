import logging
from typing import Annotated

from fastapi import Depends
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime

from api.chat_service.langgraph.state import ShareState
from api.embedding.service import EmbeddingService
from api.chat_service.langgraph.model import UserProfile

logger = logging.getLogger(__name__)

WELFARE_CATEGORY = "복지"
COLLECTION_NAME = "youth_policy_all"

embedding_service = EmbeddingService()


##################################################
# 도구 정의
##################################################
@tool
async def search_policy(query: str, runtime: ToolRuntime) -> str:
    """복지(금융·문화·예술) 분야 청년정책을 검색합니다."""

    user_role = runtime.state.get("user_role", "guest")
    user_profile: UserProfile | None = runtime.state.get("user_profile")

    search_query = query

    # user + profile 기반 맞춤 검색
    if user_role == "user" and user_profile is not None:
        profile_parts: list[str] = []

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

        if user_profile.income_level is not None:
            profile_parts.append(f"기준 중위소득 {user_profile.income_level}%")

        if profile_parts:
            search_query = f"{query} ({', '.join(profile_parts)})"

        logger.info(f"맞춤 검색(user) 쿼리: {search_query}")

    else:
        logger.info(f"기본 검색(guest) 쿼리: {search_query}")

    return await embedding_service.similarity_search_by_category(
        query=search_query,
        category=WELFARE_CATEGORY,
        collection_name=COLLECTION_NAME,
        k=5,
    )


##################################################
# Agent 클래스
##################################################
class WelfareAgent:
    """
    복지(금융·문화·예술) 청년정책 도메인 에이전트
    """

    def __init__(self, model: str = "openai:gpt-4o-mini") -> None:
        self.logger = logging.getLogger(f"{__name__}.WelfareAgent")
        self.model = model

        self.agent = create_agent(
            model=self.model,
            tools=[search_policy],
            state_schema=ShareState,  # type: ignore
            system_prompt=(
                "당신은 청년정책 중 '복지(금융·문화·예술)' 분야 전문가입니다.\n"
                "1. 반드시 search_policy 결과만 기반으로 답변하세요.\n"
                "2. 없는 내용은 추측하지 마세요.\n"
                "3. 자격요건, 신청방법, 기간을 구체적으로 설명하세요.\n"
                "4. 없으면 모른다고 말하고 재질문을 유도하세요."
            ),
        )

    async def run(
        self,
        user_query: str,
        user_role: str = "guest",
        user_profile: UserProfile | None = None,
    ) -> str:

        result = await self.agent.ainvoke(
            {
                "messages": [{"role": "user", "content": user_query}],
                "user_query": user_query,
                "user_role": user_role,
                "user_profile": user_profile,
                "category": WELFARE_CATEGORY,
                "final_response": "",
            }  # type: ignore
        )

        return result["messages"][-1].content


##################################################
# FastAPI DI
##################################################
WelfareAgentDep = Annotated[WelfareAgent, Depends(WelfareAgent)]


##################################################
# Local test
##################################################
if __name__ == "__main__":
    import asyncio
    import selectors
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parents[2]))

    from api.chat_service.langgraph.model import UserProfile

    logging.basicConfig(level=logging.INFO)
    agent = WelfareAgent()

    def _selector_loop() -> asyncio.AbstractEventLoop:
        return asyncio.SelectorEventLoop(selectors.SelectSelector())

    test_question = "저소득 청년을 위한 문화비 지원 정책 알려줘"

    # guest
    response_guest = asyncio.run(
        agent.run(test_question, user_role="guest"),
        loop_factory=_selector_loop,
    )

    print("=== guest ===")
    print(response_guest)

    # user
    test_profile = UserProfile(
        age=27,
        gender="여성",
        is_university_graduate=True,
        major="문화예술경영",
        region="울산광역시",
        income_level=150,
    )

    response_user = asyncio.run(
        agent.run(
            test_question,
            user_role="user",
            user_profile=test_profile,
        ),
        loop_factory=_selector_loop,
    )

    print("\n=== user ===")
    print(test_profile)
    print(response_user)