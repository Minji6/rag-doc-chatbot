import logging
import re
from typing import Annotated
from fastapi import Depends
from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langchain.tools import tool
from ._stub_agent import StubAgent
from ..constants import AGENT_CATEGORY

from tools.common.tool_priority import answer_with_priority
from api.chat_service.langgraph.state import DomainResult
from api.chat_service.langgraph.constants import AGENT_CATEGORY

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
    
    def _parse_policies_from_messages(self, messages: list) -> tuple[list[dict], str]:
        """
        agent의 메시지 히스토리에서 도구 호출 결과를 파싱하여 정책 메타 추출.

        새로운 포맷:
        - "[RAG]\n..." : RAG 결과 (source="rag")
        - "[LLM]\n..." : WEB 결과를 LLM 형식으로 포장 (source="web")

        Returns:
            (policies, source): 정책 메타 리스트와 출처 ("rag" / "web" / "none")
        """
        policies: list[dict] = []
        source = "none"

        for msg in messages:
            if isinstance(msg, ToolMessage):
                tool_result = msg.content

                # "[RAG]" 또는 "[LLM]" 신호로 섹션 분리
                rag_section = ""
                llm_section = ""

                # "[RAG]\n..." 추출
                if "[RAG]\n" in tool_result:
                    rag_match = re.search(r"\[RAG\]\n(.+?)(?:\n\n\[LLM\]|\Z)", tool_result, re.DOTALL)
                    if rag_match:
                        rag_section = rag_match.group(1)

                # "[LLM]\n..." 추출
                if "[LLM]\n" in tool_result:
                    llm_match = re.search(r"\[LLM\]\n(.+)", tool_result, re.DOTALL)
                    if llm_match:
                        llm_section = llm_match.group(1)

                # RAG 섹션 파싱
                if rag_section:
                    policy_blocks = [block.strip() for block in rag_section.split("\n\n") if block.strip() and "총" not in block]

                    for block in policy_blocks:
                        lines = block.split("\n")
                        if not lines:
                            continue

                        first_line = lines[0]
                        match = re.match(r"\[\d+\]\s+(.+)", first_line)
                        if match:
                            policy_name = match.group(1)
                            policy_meta = {"plcyNm": policy_name}

                            for line in lines[1:]:
                                if line.startswith("설명:"):
                                    policy_meta["plcyExplnCn"] = line.replace("설명:", "").strip()
                                elif line.startswith("지원내용:"):
                                    policy_meta["plcySprtCn"] = line.replace("지원내용:", "").strip()
                                elif line.startswith("신청방법:"):
                                    policy_meta["plcyAplyMthdCn"] = line.replace("신청방법:", "").strip()

                            policies.append(policy_meta)
                            if source == "none":
                                source = "rag"

                # LLM 섹션 파싱 (WEB 결과)
                if llm_section:
                    policy_blocks = [block.strip() for block in llm_section.split("\n\n") if block.strip()]

                    for block in policy_blocks:
                        lines = block.split("\n")
                        if not lines:
                            continue

                        first_line = lines[0]
                        match = re.match(r"\[\d+\]\s+(.+)", first_line)
                        if match:
                            policy_meta = {}

                            for line in lines:
                                if line.startswith("["):
                                    match = re.match(r"\[\d+\]\s+(.+)", line)
                                    if match:
                                        policy_meta["title"] = match.group(1)
                                elif line.startswith("출처:"):
                                    policy_meta["url"] = line.replace("출처:", "").strip()
                                elif line.startswith("요약:"):
                                    policy_meta["content"] = line.replace("요약:", "").strip()

                            if policy_meta:
                                policies.append(policy_meta)
                                source = "web"  # WEB이 있으면 source="web"으로 우선

        return policies, source
    
    async def run(self, query: str) -> DomainResult:
        """사용자 질문을 처리하고 정책 검색 결과를 DomainResult로 반환합니다."""
        try:
            result = await self.agent.ainvoke(
                {"messages": [{"role": "user", "content": query}]}
            )
            
            # agent.ainvoke 반환값에서 정보 추출
            messages = result["messages"]
            text = messages[-1].content if messages else ""
            
            # 메시지 히스토리에서 정책 메타 파싱
            policies, source = self._parse_policies_from_messages(messages)
            
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