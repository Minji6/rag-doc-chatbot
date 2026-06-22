import logging
from typing import Annotated
from fastapi import Depends
from langchain.agents import create_agent
from ..constants import AGENT_CATEGORY
from tools.common.tool_compare import compare_policies, policy_priority_score
from tools.common.tool_dday import calculate_dday
from tools.welfare.tool_eligibility import check_eligibility
from tools.welfare.tool_user_profile import build_profile_context

logger = logging.getLogger(__name__)

_CATEGORY = AGENT_CATEGORY["welfare"]

_SYSTEM_PROMPT = """당신은 청년 복지문화 정책 전문가입니다.
제공된 [정책 정보]에만 근거하여 답변하세요. 정보에 없는 정책이나 수치를 임의로 만들어내지 마세요.
[정책 정보]가 비어 있으면, 관련 정책을 찾지 못했다고 솔직하게 안내하세요.

## 도구 사용 규칙

### calculate_dday (마감일 관련 질문 시)
- 사용자가 마감일을 물을 때 호출하세요.
- deadline: bizPrdEndYmd 값, apply_period_type: aplyPrdSeCd 값을 사용하세요.
- 마감일 데이터가 없으면 호출하지 마세요.

### compare_policies (비교 요청 시)
- 사용자가 "비교해줘"라고 하면 호출하세요.
- policy_list에 [정책 정보]에서 파악한 정책 딕셔너리 목록을 전달하세요.

### policy_priority_score (추천 요청 시)
- 사용자가 "추천해줘"라고 하면 호출하세요.
- policy_list에 [정책 정보]에서 파악한 정책 딕셔너리 목록을 전달하세요.

### check_eligibility (자격 확인 요청 시)
- 사용자가 "신청 가능해?", "자격 되나?" 등을 물을 때만 호출하세요.
- policy_metadata에 해당 정책의 정보를 딕셔너리로 전달하세요.

## 사용자 프로필 안내 지침
- [사용자 프로필]에 정보가 있으면 프로필에 맞는 정책을 우선 안내하세요.
- guest(비로그인)인 경우 일반적인 복지 정책을 안내하세요."""


class WelfareAgent:
    """
    복지문화 Agent — 생성(generation) 전용.

    정책 검색은 welfare_search_node가 담당하고 state에 기록한다.
    이 에이전트는 검색된 정책 텍스트(knowledge)를 받아 답변만 생성한다.
    (검색/생성 분리 구조)
    """

    def __init__(self, model: str = "openai:gpt-4o-mini") -> None:
        self.logger = logging.getLogger(f"{__name__}.WelfareAgent")
        self._model = model

    def _make_agent(self, user_role: str, user_profile: dict):
        system_prompt = _SYSTEM_PROMPT
        profile_context = build_profile_context(user_role, user_profile)
        if profile_context:
            system_prompt = f"{system_prompt}\n\n{profile_context}"
        return create_agent(
            model=self._model,
            tools=[
                calculate_dday,
                compare_policies,
                policy_priority_score,
                check_eligibility,
            ],
            system_prompt=system_prompt,
        )

    async def run(
        self,
        inquiry: str,
        knowledge: str,
        user_role: str = "guest",
        user_profile: dict | None = None,
    ) -> str:
        """검색된 정책(knowledge)으로 답변을 생성한다. (검색은 하지 않음)

        Args:
            inquiry: 사용자 질문
            knowledge: welfare_search_node가 검색한 정책 텍스트
            user_role: "guest" 또는 "user"
            user_profile: 로그인 유저 프로필 (guest면 None/빈 dict)
        Returns:
            str: 생성된 사용자용 답변 텍스트
        """
        user_profile = user_profile or {}
        agent = self._make_agent(user_role, user_profile)

        prompt = (
            "다음 정보를 바탕으로 복지문화 정책 답변을 작성하세요.\n\n"
            f"[질문]\n{inquiry}\n\n"
            f"[정책 정보]\n{knowledge or '(검색된 정책 없음)'}\n"
        )

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )
        return result["messages"][-1].content


WelfareAgentDep = Annotated[WelfareAgent, Depends(WelfareAgent)]
