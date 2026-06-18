import logging
from typing import Annotated

from fastapi import Depends
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_postgres import PGVector
from langchain.embeddings import init_embeddings

from api.chat_service.langgraph.model import UserProfile
from api.chat_service.langgraph.state import ShareState

logger = logging.getLogger(__name__)


##############################################################
# VECTORSTORE (임시로 여기 포함)
##############################################################
vectorstore = PGVector(
    embeddings=init_embeddings(model="openai:text-embedding-3-large"),
    collection_name="youth_policy_all",
    connection=None,  # ← 너 engine 넣어줘야 함 (기존 코드 기준 유지)
    async_mode=True,
)

def build_profile_context(profile) -> str:
    parts = []

    basic = profile.basic
    economic = profile.economic
    education = profile.education
    special = profile.special

    if basic.age:
        parts.append(f"{basic.age}세")

    if basic.gender:
        parts.append(basic.gender)

    if basic.region:
        parts.append(basic.region)

    if education.education_level:
        parts.append(education.education_level)

    if education.major:
        parts.append(education.major)

    if economic.income_level:
        parts.append(f"소득 {economic.income_level}%")

    if economic.employment_status:
        parts.append(economic.employment_status)

    if special.is_low_income:
        parts.append("저소득")

    if special.company_type:
        parts.append(special.company_type)

    return ", ".join(parts)

@tool
async def search_policy(query: str, runtime: ToolRuntime) -> str:
    user_role = runtime.state.get("user_role", "guest")
    profile_context = runtime.state.get("profile_context", "")

    search_query = query

    if user_role == "user" and profile_context:
        search_query = f"{query} ({profile_context})"

    results = await vectorstore.asimilarity_search_with_score(
        search_query,
        k=5,
        filter={"category": "복지문화"},
    )

    documents = [(doc, dist) for doc, dist in results if dist < 0.4]

    if not documents:
        return f"'{search_query}' 관련 복지 정책을 찾을 수 없습니다."

    return "\n".join(
        f"[정책 {i}]\n{doc.metadata.get('title','')}\n{doc.page_content}\n"
        for i, (doc, _) in enumerate(documents, 1)
    )

class WelfareAgent:
    """
    청년 복지(금융·문화·예술) 정책 에이전트
    """

    def __init__(self, model: str = "openai:gpt-4o-mini") -> None:
        self.logger = logging.getLogger(f"{__name__}.WelfareAgent")

        self.agent = create_agent(
            model=model,
            tools=[search_policy],
            state_schema=ShareState,  # type: ignore
            system_prompt=(
                "당신은 청년 복지(금융·문화·예술) 정책 전문가입니다. "
                "복지 정책만 정확하게 안내하세요."
            ),
        )

    async def run(
        self,
        question: str,
        user_role: str = "guest",
        user_profile=None,
    ) -> str:

        # dict → Pydantic 변환
        if isinstance(user_profile, dict):
            user_profile = UserProfile.model_validate(user_profile)

        # profile context 생성
        profile_context = ""
        if user_role == "user" and user_profile:
            profile_context = build_profile_context(user_profile)

        result = await self.agent.ainvoke(
            {
                "messages": [
                    {"role": "user", "content": question}
                ],
                "user_role": user_role,
                "profile_context": profile_context,
            }
        )

        return result["messages"][-1].content
    
WelfareAgentDep = Annotated[WelfareAgent, Depends(WelfareAgent)]