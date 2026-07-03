import json
import logging
from typing import Annotated
from fastapi import Depends
from langchain.agents import create_agent
from ..constants import AGENT_CATEGORY, OUTPUT_FORMAT_GUIDE, COMPARISON_COMMENT_GUIDE, OUTPUT_DETAIL_GUIDE, RECOMMEND_GUIDE
from ..tools import SUGGESTIONS_PROMPT, parse_suggestions, calculate_dday

logger = logging.getLogger(__name__)

_CATEGORY = AGENT_CATEGORY["employment"]

_TOOL_USAGE_GUIDE = """[도구 사용 규칙 — 최우선, 아래 출력 형식 규칙보다 우선한다]
- 사용자가 마감일·신청 기간을 물으면 calculate_dday를 호출하세요.
  deadline에는 bizPrdEndYmd 값을, apply_period_type에는 aplyPrdSeCd 값을 전달하세요.
- 도구를 호출했다면 그 결과를 답변 맨 앞 "### 마감일 안내" 블록으로
  먼저 제시한 뒤, 정책 목록을 이어 붙이세요. 도구 결과를 추측으로 대체하지 마세요.

"""


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
            tools=[calculate_dday],
            system_prompt=_TOOL_USAGE_GUIDE + """당신은 청년 일자리 정책 전문가입니다.
                제공된 [정책 정보]에만 근거하여 답변하세요. 정보에 없는 정책이나 수치를 임의로 만들어내지 마세요.
                [정책 정보]가 비어 있으면, 관련 정책을 찾지 못했다고 솔직하게 안내하세요.

                [중요 규칙]
                각 정책의 신청기간 항목에 "⚠️ 현재 신청 불가"가 포함되어 있으면,
                해당 정책 안내 마지막에 "※ 현재 신청기간이 종료된 정책입니다. 향후 재공고를 확인하세요."를 반드시 추가하세요.

                # 답변 형식 (인사말·맺음말 없이 곧바로 아래 본문부터)
                정책마다 아래 ### 블록을 반복하세요.

                ### [정책명]
                - 개요: 정책 설명 / 지원 규모 및 대상
                - 지원 내용: 지원 유형별 세부 내용 / 지원 금액·범위
                - 참여 자격: 기본 요건 / 연령·고용 상태 조건 / 업종·직종 제한 여부
                - 신청 방법: 신청 기간·경로 확인 → 필수 서류 준비 → 기관(고용센터 등) 신청 → 자격 심사 및 선정
                - 신청 URL: 정책 정보의 "신청 URL" 값을 raw URL 그대로 (없으면 "정보 없음")

                [필수 확인사항]
                ✓ 신청 마감일 반드시 확인
                ✓ 지원 대상 연령/고용 상태 확인
                ✓ 필요 서류 사전 준비""" + OUTPUT_FORMAT_GUIDE
        )

    async def run(
        self, inquiry: str, knowledge: str, policies: list[dict] | None = None,
        user_profile: dict | None = None,
        inquiry_type: str = "검색"
    ) -> tuple[str, list[str]]:
        """검색된 정책(knowledge)으로 답변을 생성한다. (검색은 하지 않음)

        Args:
            inquiry: 사용자 질문
            knowledge: employment_search_node가 검색한 정책 텍스트
            policies: employment_search_node가 검색한 raw 정책 메타 (도구 호출용 구조화 데이터)
            user_profile: 로그인 유저 프로필 (guest면 None/빈 dict)
            inquiry_type: 질문 의도. "비교"면 표 대신 짧은 정성 코멘트만 생성.
        Returns:
            tuple[str, list[str]]: (생성된 답변 텍스트, follow-up 질문 목록)
        """
        policies = policies or []

        prompt = (
            "다음 정보를 바탕으로 일자리 정책 답변을 작성하세요.\n\n"
            f"[질문]\n{inquiry}\n\n"
            f"[정책 정보]\n{knowledge or '(검색된 정책 없음)'}\n"
        )
        if user_profile:
            prompt += f"\n[사용자 정보]\n{user_profile}\n"

        # calculate_dday 도구가 정책 딕셔너리를 그대로 받을 수 있도록 원본 메타를 첨부.
        if policies:
            prompt += (
                f"\n[정책 구조화 데이터]\n"
                f"{json.dumps(policies, ensure_ascii=False, indent=2)}\n"
                "마감일 도구 호출 시 위 데이터를 사용하세요.\n"
            )

        # 비교 모드: 정형 표는 composer 담당. 에이전트는 분야 관점 코멘트만.
        if inquiry_type == "비교":
            prompt += COMPARISON_COMMENT_GUIDE
        # 상세조회: 특정 정책 1개 깊이 안내.
        elif inquiry_type == "상세조회":
            prompt += OUTPUT_DETAIL_GUIDE
        # 추천: 각 정책 블록에 추천 이유 + 적합도 점수를 노출.
        elif inquiry_type == "추천":
            prompt += RECOMMEND_GUIDE

        prompt += SUGGESTIONS_PROMPT
        result = await self.agent.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )
        return parse_suggestions(result["messages"][-1].content)


EmploymentAgentDep = Annotated[EmploymentAgent, Depends(EmploymentAgent)]
