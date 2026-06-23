import logging
from typing import Annotated
from fastapi import Depends
from langchain.agents import create_agent

logger = logging.getLogger(__name__)

##############################################################
# Agent 클래스 정의
##############################################################
class HousingAgent:
    def __init__(self, model: str = "openai:gpt-4o-mini") -> None:
        self.logger = logging.getLogger(f"{__name__}.HousingAgent")
        self.agent = create_agent(
            model=model,
            tools=[],
            system_prompt="""당신은 청년 주거 정책 전문가입니다.
             제공된 [정책 정보]에만 근거하여 답변하세요. 정보에 없는 정책이나 수치를 임의로 만들어내지 마세요.
                [정책 정보]가 비어 있으면, 관련 정책을 찾지 못했다고 솔직하게 안내하세요.

                답변 형식은 다음과 같습니다.

                🏠 주거정책팀 답변:
                안녕하세요, 청년 주거정책팀입니다.

                [정책명 및 개요]
                - 정책 설명
                - 지원 규모 및 대상

                [지원 내용]
                - 지원 유형(전세대출/월세지원/공공임대 등)
                - 지원 금액 및 한도

                [참여 대상 자격]
                - 연령/소득/거주지 요건
                - 무주택 여부 등 추가 조건

                [신청 방법 및 절차]
                1. 신청 기간 및 경로 확인
                2. 필수 서류 준비
                3. 온라인/오프라인 접수
                4. 자격 심사 및 최종 선정

                [필수 확인사항]
                ✓ 신청 마감일 반드시 확인
                ✓ 소득/자산 기준 사전 점검
                ✓ 필요 서류 사전 준비

                친절하게 안내해드리겠습니다.
                감사합니다.""",
        )

    async def run(
        self, inquiry: str, knowledge: str, user_profile: dict | None = None) -> str:
        
        prompt = (
            "다음 정보를 바탕으로 주거 정책 답변을 작성하세요. \n\n"
            f"[질문\n{inquiry}]\n\n"
            f"[정책정보]\n{knowledge or '(검색한 정책 없음)'}\n"
        )
        
        if user_profile:
            prompt += f"\n[사용자 정보]\n{user_profile}\n"
        
        result = await self.agent.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )
        
        return result["messages"][-1].content


HousingAgentDep = Annotated[HousingAgent, Depends(HousingAgent)]
