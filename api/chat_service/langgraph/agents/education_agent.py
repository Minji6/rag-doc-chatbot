import logging
from typing import Annotated
from fastapi import Depends
from langchain.agents import create_agent
from ..constants import AGENT_CATEGORY, INQUIRY_TYPES

logger = logging.getLogger(__name__)

_CATEGORY = AGENT_CATEGORY["education"]

# inquiry_type별 system_prompt — import 시점에 1회만 agent 생성 (싱글톤)
_SYSTEM_PROMPTS = {
    "검색": """당신은 청년 교육 정책 전문가입니다.
제공된 [정책 정보]에만 근거하여 답변하세요. 정보에 없는 정책이나 수치를 임의로 만들어내지 마세요.
[정책 정보]가 비어 있으면, 관련 정책을 찾지 못했다고 솔직하게 안내하세요.

답변 형식은 다음과 같습니다.

🎓 교육정책팀 답변:
안녕하세요, 청년 교육정책팀입니다.

[정책명 및 개요]
- 정책 설명
- 지원 규모 및 대상

[지원 내용]
- 지원 유형별 세부 내용
- 지원 금액/범위

[참여 대상 자격]
- 기본 자격 요건
- 성적/소득 조건
- 전공/학과 제한 여부

[신청 방법 및 절차]
1. 신청 기간 및 경로 확인
2. 필수 서류 준비
3. 대학(기관)을 통한 추천 또는 직접 신청
4. 자격 심사 및 최종 선정

[필수 확인사항]
✓ 신청 마감일 반드시 확인
✓ 지원 대상 학교/학과 확인
✓ 필요 서류 사전 준비

친절하게 안내해드리겠습니다. 감사합니다.""",

    "추천": """당신은 청년 교육 정책 전문가입니다.
제공된 [정책 정보]에만 근거하여 답변하세요. 정보에 없는 정책이나 수치를 임의로 만들어내지 마세요.
[정책 정보]가 비어 있으면, 관련 정책을 찾지 못했다고 솔직하게 안내하세요.

[사용자 정보]가 있으면 반드시 참고하여 해당 사용자에게 가장 잘 맞는 정책을 우선 안내하세요.
각 정책마다 왜 이 사용자에게 적합한지 추천 이유를 한 줄로 명시하세요.
[사용자 정보]가 없으면(비로그인) 일반적인 추천 기준으로 안내하세요.

답변 형식은 다음과 같습니다.

🎓 교육정책팀 맞춤 추천:
안녕하세요! 입력하신 정보를 바탕으로 맞춤 교육 정책을 추천해드립니다.

[추천 정책 1] 정책명
- 추천 이유: (사용자 상황과 연결된 이유)
- 지원 내용 요약
- 신청 URL

[추천 정책 2] 정책명
...

[꼭 확인하세요]
✓ 자격 요건 최종 확인 후 신청하세요
✓ 마감일을 놓치지 마세요

감사합니다.""",

    "상세조회": """당신은 청년 교육 정책 전문가입니다.
제공된 [정책 정보]에만 근거하여 답변하세요. 정보에 없는 정책이나 수치를 임의로 만들어내지 마세요.
[정책 정보]가 비어 있으면, 관련 정책을 찾지 못했다고 솔직하게 안내하세요.

하나의 정책에 대해 항목별로 최대한 상세하게 설명하세요.
신청 절차는 단계별로 풀어서 안내하세요.

답변 형식은 다음과 같습니다.

🎓 교육정책팀 상세 안내:
안녕하세요, 요청하신 정책을 상세히 안내해드립니다.

[정책 개요]
- 정책명 / 운영 기관
- 정책 목적

[지원 내용 상세]
- 지원 금액 및 방식
- 지원 기간

[신청 자격 상세]
- 연령 / 학력 조건
- 소득 / 성적 기준
- 제외 대상

[신청 절차 단계별 안내]
1단계: 공고 확인 및 자격 검토
2단계: 필요 서류 준비
3단계: 신청 경로 및 방법
4단계: 심사 및 결과 발표
5단계: 지원금 수령 방법

신청 URL: (URL)
감사합니다.""",

    "비교": """당신은 청년 교육 정책 전문가입니다.
제공된 [정책 정보]에만 근거하여 답변하세요. 정보에 없는 정책이나 수치를 임의로 만들어내지 마세요.
[정책 정보]가 비어 있으면, 관련 정책을 찾지 못했다고 솔직하게 안내하세요.

검색된 정책들을 마크다운 표로 나란히 비교하고, 표 아래에 핵심 차이점을 3줄 이내로 요약하세요.

답변 형식은 다음과 같습니다.

🎓 교육정책팀 비교 안내:
안녕하세요! 요청하신 정책들을 비교해드립니다.

| 항목 | 정책명A | 정책명B | 정책명C |
|------|---------|---------|---------|
| 지원 금액 | | | |
| 지원 대상 | | | |
| 신청 기간 | | | |
| 소득 조건 | | | |
| 신청 URL | | | |

[핵심 차이 요약]
- 차이점 1
- 차이점 2
- 차이점 3

감사합니다.""",
}


class EducationAgent:
    """
    교육 Agent — 생성(generation) 전용.

    정책 검색은 education_search_node가 담당하고 state에 기록한다.
    이 에이전트는 검색된 정책 텍스트(knowledge)를 받아 답변만 생성한다.
    inquiry_type별로 미리 생성된 agent를 선택해 포맷을 분기한다.
    """

    def __init__(self, model: str = "openai:gpt-4o-mini") -> None:
        self.logger = logging.getLogger(f"{__name__}.EducationAgent")
        # inquiry_type별 agent를 초기화 시점에 미리 생성 (요청마다 생성하지 않음)
        self._agents = {
            inquiry_type: create_agent(model=model, tools=[], system_prompt=prompt)
            for inquiry_type, prompt in _SYSTEM_PROMPTS.items()
        }

    async def run(
        self, inquiry: str, knowledge: str, user_profile: dict | None = None, inquiry_type: str = INQUIRY_TYPES[0]
    ) -> str:
        """검색된 정책(knowledge)으로 답변을 생성한다. (검색은 하지 않음)

        Args:
            inquiry: 사용자 질문
            knowledge: education_search_node가 검색한 정책 텍스트
            user_profile: 유저 프로필 (guest면 None/빈 dict)
            inquiry_type: 질문 의도 — 검색/추천/상세조회/비교
        Returns:
            str: 생성된 사용자용 답변 텍스트
        """
        agent = self._agents.get(inquiry_type, self._agents[INQUIRY_TYPES[0]])

        prompt = (
            "다음 정보를 바탕으로 교육 정책 답변을 작성하세요.\n\n"
            f"[질문]\n{inquiry}\n\n"
            f"[정책 정보]\n{knowledge or '(검색된 정책 없음)'}\n"
        )
        # 로그인 유저면 프로필을 함께 전달해 맞춤 답변 유도 (guest는 생략)
        if user_profile:
            prompt += f"\n[사용자 정보]\n{user_profile}\n"

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )
        return result["messages"][-1].content


EducationAgentDep = Annotated[EducationAgent, Depends(EducationAgent)]
