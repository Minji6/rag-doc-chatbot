import logging
from typing import Annotated
from fastapi import Depends
from langchain.agents import create_agent
from ..constants import AGENT_CATEGORY

logger = logging.getLogger(__name__)

_CATEGORY = AGENT_CATEGORY["employment"]


class EmploymentAgent:
    """
    취업 Agent — 생성(generation) 전용.

    정책 검색은 employment_search_node가 담당하고 state에 기록한다.
    이 에이전트는 검색된 정책 텍스트(knowledge)를 받아 답변만 생성한다.
    (검색/생성 분리 구조 — 교수님 sec09 멀티에이전트 패턴)
    """

    def __init__(self, model: str = "openai:gpt-4o-mini") -> None:
        self.logger = logging.getLogger(f"{__name__}.EmploymentAgent")
        self.agent = create_agent(
            model=model,
            tools=[],  # 검색 도구 없음 — 생성 전용
            system_prompt="""당신은 청년 일자리 정책 전문가입니다.
                제공된 [정책 정보]에만 근거하여 답변하세요. 정보에 없는 정책이나 수치를 임의로 만들어내지 마세요.
                [정책 정보]가 비어 있으면, 관련 정책을 찾지 못했다고 솔직하게 안내하세요.

                답변 형식은 다음과 같습니다.

                💼 일자리정책팀 답변:
                안녕하세요, 청년 일자리정책팀입니다.

                [정책명 및 개요]
                - 정책 설명
                - 지원 규모 및 대상

                [지원 내용]
                - 지원 유형별 세부 내용
                - 지원 금액/범위

                [참여 대상 자격]
                - 기본 자격 요건
                - 연령/고용 상태 조건
                - 업종/직종 제한 여부

                [신청 방법 및 절차]
                1. 신청 기간 및 경로 확인
                2. 필수 서류 준비
                3. 기관(고용센터 등)을 통한 신청
                4. 자격 심사 및 최종 선정

                [필수 확인사항]
                ✓ 신청 마감일 반드시 확인
                ✓ 지원 대상 연령/고용 상태 확인
                ✓ 필요 서류 사전 준비

                친절하게 안내해드리겠습니다.
                감사합니다."""
        )

    async def run(
        self, inquiry: str, knowledge: str, user_profile: dict | None = None
    ) -> str:
        """검색된 정책(knowledge)으로 답변을 생성한다. (검색은 하지 않음)

        Args:
            inquiry: 사용자 질문
            knowledge: employment_search_node가 검색한 정책 텍스트
            user_profile: 로그인 유저 프로필 (guest면 None/빈 dict)
        Returns:
            str: 생성된 사용자용 답변 텍스트
        """
        prompt = (
            "다음 정보를 바탕으로 일자리 정책 답변을 작성하세요.\n\n"
            f"[질문]\n{inquiry}\n\n"
            f"[정책 정보]\n{knowledge or '(검색된 정책 없음)'}\n"
        )
        if user_profile:
            prompt += f"\n[사용자 정보]\n{user_profile}\n"

        result = await self.agent.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )
        return result["messages"][-1].content


EmploymentAgentDep = Annotated[EmploymentAgent, Depends(EmploymentAgent)]
