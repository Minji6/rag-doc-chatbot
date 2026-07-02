import json
import logging
from typing import Annotated
from fastapi import Depends
from langchain.agents import create_agent

from ..constants import OUTPUT_FORMAT_GUIDE, COMPARISON_COMMENT_GUIDE, OUTPUT_DETAIL_GUIDE, RECOMMEND_GUIDE
from ..tools import SUGGESTIONS_PROMPT, parse_suggestions, calculate_dday, check_eligibility_detailed

logger = logging.getLogger(__name__)

# 검색 노드가 이미 나이/지역/혼인 기준으로 ✅/❌/❓을 매겨두지만, 사용자가 특정 정책을 콕 집어
# "신청 가능해?"처럼 물으면 취업/소득/학력까지 포함한 상세 판정이 필요하다 — 그때만 도구를 쓴다.
_TOOL_USAGE_GUIDE = """[도구 사용 규칙 — 최우선, 아래 출력 형식 규칙보다 우선한다]
- 사용자가 "신청 가능해?", "자격 되나?", "해당돼?" 등 특정 정책의 자격을 물으면,
  정책을 나열하기 전에 먼저 check_eligibility_detailed를 호출하세요.
  user_profile에는 [사용자 정보] 값을, policy_metadata에는 [정책 구조화 데이터]에서 해당 정책 딕셔너리를
  그대로 전달하세요. [정책 구조화 데이터]에 있는 정책에만 사용하세요.
- 사용자가 마감일·신청 기간을 물으면 calculate_dday를 호출하세요.
  deadline에는 bizPrdEndYmd 값을, apply_period_type에는 aplyPrdSeCd 값을 전달하세요.
- 도구를 호출했다면 그 결과를 답변 맨 앞 "### 자격 판정 결과"(또는 "### 마감일 안내") 블록으로
  먼저 제시한 뒤, 정책 목록을 이어 붙이세요. 도구 결과를 추측으로 대체하지 마세요.

"""

##############################################################
# Agent 클래스 정의
##############################################################


class HousingAgent:
    def __init__(self, model: str = "openai:gpt-4o-mini") -> None:
        self.logger = logging.getLogger(f"{__name__}.HousingAgent")
        self.agent = create_agent(
            model=model,
            tools=[calculate_dday, check_eligibility_detailed],
            system_prompt=_TOOL_USAGE_GUIDE + """당신은 청년 주거 정책 전문가입니다.

            제공된 [정책 정보]에만 근거하여 답변하세요. 정보에 없는 정책이나 수치를 임의로 만들어내지 마세요.
            [정책 정보]가 비어 있으면, 관련 정책을 찾지 못했다고 솔직하게 안내하세요.

            # 자격 진단 표시 처리 규칙 (이 규칙 자체는 절대 답변에 포함하지 말 것)

            각 정책 옆에 ✅/❌/❓ 표시가 있다면 사용자 프로필 기반 사전 진단 결과입니다.
            - ✅: 자격 적합 → 【추천 정책】 섹션에 모두 안내. 1개여도 여러 개여도 전부 ### 블록으로 나열.
            - ❓: 추가 확인 필요 → 【추가 확인 필요】 섹션에 정책명 + 한 줄 사유만 짧게.
            - ❌: 자격 미충족 → 【자격 미충족 안내】 섹션에 정책명만 한 줄로 나열.
            - 진단 표시가 없으면(=게스트) 자격 언급 없이 ### 블록 형식으로만 일반 안내.
            - 해당 카테고리(✅/❓/❌)에 정책이 0개면 그 섹션 자체를 출력하지 마세요.

            # 답변 형식 (인사말·맺음말 없이 곧바로 아래 본문부터)

            【추천 정책】
            ### [정책명]
            - 개요: 정책 설명 + 지원 규모/대상
            - 지원 내용: 지원 유형(전세대출/월세지원/공공임대 등), 금액 및 한도
            - 참여 자격: 연령/소득/거주지/혼인/무주택 요건
            - 신청 방법: 신청 기간, 경로, 절차 요약
            - 신청 URL: 제공된 URL (없으면 "정보 없음")

            (✅ 정책이 여러 개면 위 ### 블록을 정책 수만큼 반복)

            【추가 확인 필요】
            - [정책명]: 한 줄 사유

            【자격 미충족 안내】
            다음 정책들은 자격 요건이 맞지 않아 상세 안내에서 제외했습니다: [정책명], [정책명] ...

            [필수 확인사항]
            ✓ 신청 마감일 반드시 확인
            ✓ 소득/자산 기준 사전 점검
            ✓ 필요 서류 사전 준비""" + OUTPUT_FORMAT_GUIDE,
        )

    async def run(
            self, inquiry: str, knowledge: str, policies: list[dict] | None = None,
            user_profile: dict | None = None,
            inquiry_type: str = "검색") -> tuple[str, list[str]]:

        policies = policies or []

        # ✅ 라벨이 붙은 정책 개수 카운트
        eligible_count = knowledge.count("✅ 자격 적합")
        ineligible_count = knowledge.count("❌ 자격 미충족")

        prompt = (
            "다음 정보를 바탕으로 주거 정책 답변을 작성하세요. \n\n"
            f"[질문]\n{inquiry}\n\n"
            f"[정책정보]\n{knowledge or '(검색한 정책 없음)'}\n"
        )

        if user_profile:
            prompt += f"\n[사용자 정보]\n{user_profile}\n"

        # check_eligibility_detailed/calculate_dday 도구가 정책 딕셔너리를 그대로 받을 수 있도록 원본 메타를 첨부.
        if policies:
            prompt += (
                f"\n[정책 구조화 데이터]\n"
                f"{json.dumps(policies, ensure_ascii=False, indent=2)}\n"
                "자격 확인·마감일 도구 호출 시 위 데이터를 사용하세요.\n"
            )

        # 비교 모드: 표는 composer가 그리므로 ### 블록 강제 지시 대신 짧은 정성 코멘트만 요청.
        if inquiry_type == "비교":
            prompt += COMPARISON_COMMENT_GUIDE
            prompt += SUGGESTIONS_PROMPT
            result = await self.agent.ainvoke(
                {"messages": [{"role": "user", "content": prompt}]}
            )
            return parse_suggestions(result["messages"][-1].content)

        # 강제 지시 (시스템 프롬프트보다 강력)
        if eligible_count > 0:
            prompt += (
                f"\n\n[필수 준수] 정책 정보에 ✅ 자격 적합 정책이 정확히 "
                f"{eligible_count}개 있습니다. 【추천 정책】 섹션에 반드시 "
                f"{eligible_count}개 모두 ▶ 블록으로 나열하세요. "
                f"1개만 안내하면 답변이 무효입니다.\n"
            )
        if ineligible_count > 0:
            prompt += (
                f"❌ 자격 미충족 정책이 {ineligible_count}개 있습니다. "
                f"【자격 미충족 안내】 섹션에 모두 나열하세요.\n"
            )

        # 상세조회: 특정 정책 1개를 깊이 있게 안내하도록 유도.
        if inquiry_type == "상세조회":
            prompt += OUTPUT_DETAIL_GUIDE
        # 추천: 각 정책 블록에 추천 이유 + 적합도 점수를 노출.
        if inquiry_type == "추천":
            prompt += RECOMMEND_GUIDE

        prompt += SUGGESTIONS_PROMPT
        result = await self.agent.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )

        return parse_suggestions(result["messages"][-1].content)


HousingAgentDep = Annotated[HousingAgent, Depends(HousingAgent)]
