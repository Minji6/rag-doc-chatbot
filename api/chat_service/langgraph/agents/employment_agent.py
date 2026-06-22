import logging
import re
import json
from typing import Annotated
from fastapi import Depends
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from ._stub_agent import StubAgent
from ..constants import AGENT_CATEGORY

from tools.common.tool_priority import answer_with_priority
from api.chat_service.langgraph.state import DomainResult
from api.chat_service.langgraph.constants import AGENT_CATEGORY

logger = logging.getLogger(__name__)

class EmploymentAgent:
    def __init__(self, model: str = "openai:gpt-4o-mini") -> None:
        self.logger = logging.getLogger(f"{__name__}.EmploymentAgent")
        self.category = AGENT_CATEGORY["employment"]  # "일자리"

        self.agent = create_agent(
            model=model,
            tools=[answer_with_priority],
            system_prompt=f"""
당신은 청년 일자리 정책 전문 에이전트입니다.

[핵심 역할]
answer_with_priority 도구를 호출하여 정책을 검색하고 답변합니다.
이 도구는 내부적으로:
1. PGVector RAG에서 정책 검색 (1순위)
2. RAG 결과가 부족하면 웹 검색으로 보완
3. RAG와 WEB 결과를 정책 정보 형식으로 통합

[도구 호출 방법]
answer_with_priority(query, category, k)
- query: 사용자의 질문
- category: "{self.category}" (반드시 이 값 사용)
- k: 최대 정책 개수 (기본 5)

[답변 가이드]
- 정책명, 설명, 지원내용, 신청방법을 명확하게 제시
- 웹 검색 결과는 출처(URL)를 명시
- 정책 출처(RAG/WEB)를 구분하여 사용자에게 전달
- 불확실한 정보는 "확인이 필요합니다"라고 명시
            """
        )

    def _parse_policies_from_tool_result(self, tool_result: str) -> list[dict]:
        """
        tool 반환값(JSON)에서 structured metadata를 파싱합니다.
        tool_result: answer_with_priority에서 반환한 JSON 문자열
        """
        self.logger.info(f"[DEBUG] tool_result 타입: {type(tool_result)}, 길이: {len(str(tool_result))}")
        self.logger.info(f"[DEBUG] tool_result[:200]: {str(tool_result)[:200]}")

        try:
            data = json.loads(tool_result)
            policies = data.get("policies", [])
            self.logger.info(f"[DEBUG] JSON 파싱 성공, policies={len(policies)}건")
            return policies
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.logger.warning(f"[DEBUG] tool 반환값 파싱 실패: {str(e)}")
            self.logger.warning(f"[DEBUG] 전체 tool_result: {tool_result}")
            return []

    async def run(self, query: str) -> DomainResult:
        """사용자 질문을 처리하고 정책 검색 결과를 DomainResult로 반환합니다."""
        try:
            result = await self.agent.ainvoke(
                {"messages": [{"role": "user", "content": query}]}
            )

            # agent.ainvoke 반환값에서 정보 추출
            messages = result["messages"]
            text = messages[-1].content if messages else ""

            # tool 메시지에서 structured metadata 추출
            policies = []
            for msg in messages:
                if isinstance(msg, ToolMessage):
                    policies = self._parse_policies_from_tool_result(msg.content)
                    if policies:
                        break

            # 소스 판정
            source = "rag" if policies else "none"

            self.logger.info(
                f"일자리 에이전트 완료 - source={source}, policies={len(policies)}건"
            )

            return DomainResult(
                text=text,
                policies=policies,
                category=self.category,
                source=source
            )

        except Exception as e:
            self.logger.error(f"일자리 에이전트 실행 실패: {e}")
            return DomainResult(
                text=f"죄송합니다. 일자리 정책 검색 중 오류가 발생했습니다: {str(e)}",
                policies=[],
                category=self.category,
                source="none"
            )

EmploymentAgentDep = Annotated[EmploymentAgent, Depends(EmploymentAgent)]