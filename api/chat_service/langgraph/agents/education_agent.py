import logging
from contextvars import ContextVar
from typing import Annotated
from fastapi import Depends
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_postgres import PGVector
from langchain.embeddings import init_embeddings
from api.common.sqlalchemy_conf import engine
from ..state import DomainResult, empty_domain_result
from ..constants import (
    AGENT_CATEGORY,
    PGVECTOR_COLLECTION_NAME,
    POLICY_METADATA_FIELDS,
    SIMILARITY_DISTANCE_THRESHOLD,
)

logger = logging.getLogger(__name__)

_CATEGORY = AGENT_CATEGORY["education"]


def _pick_policy_fields(metadata: dict) -> dict:
    """PGVector 메타에서 화이트리스트 필드만 추려 dict 생성."""
    return {key: metadata.get(key) for key in POLICY_METADATA_FIELDS}


##############################################################
# 도구 정의
##############################################################
# search_policy의 raw 정책 결과를 노드 레벨로 회수하기 위한 채널.
_last_search_policies: ContextVar[list[dict]] = ContextVar(
    "education_last_search_policies", default=[]
)

@tool
async def search_policy(query: str) -> str:
    """
    교육 분야 청년 정책을 PGVector에서 검색합니다.
    Args:
        query: 검색할 질문이나 키워드
    Returns:
        str: 검색된 정책 내용
    """
    vectorstore = PGVector(
        embeddings=init_embeddings(model="openai:text-embedding-3-large"),
        collection_name=PGVECTOR_COLLECTION_NAME,
        connection=engine,
        async_mode=True,
    )
    results = await vectorstore.asimilarity_search_with_score(
        query, k=5, filter={"category": _CATEGORY}
    )
    documents = [(doc, dist) for doc, dist in results if dist <= SIMILARITY_DISTANCE_THRESHOLD]

    if not documents:
        _last_search_policies.set([])
        return f"'{query}' 관련 {_CATEGORY} 정책을 찾을 수 없습니다."

    _last_search_policies.set([_pick_policy_fields(doc.metadata) for doc, _ in documents])

    lines = []
    for idx, (doc, _dist) in enumerate(documents, 1):
        lines.append(f"[정책 {idx}] {doc.metadata.get('plcyNm', '')}")
        lines.append(f"내용: {doc.page_content}")
        lines.append(f"신청 URL: {doc.metadata.get('aplyUrlAddr', '정보 없음')}\n")
    return "\n".join(lines)


##############################################################
# Agent 클래스 정의
##############################################################
class EducationAgent:
    """
    교육 Agent
    교육 관련 문의(직업교육 등의 미래역량강화, 장학금, 등록금 등의 금전 지원 등)에 대한 답변을 제공합니다.
    """
    
    # 초기화 메소드    
    def __init__(self, model: str = "openai:gpt-4o-mini") -> None:
        self.logger = logging.getLogger(f"{__name__}.EducationAgent")
        self.agent = create_agent(
            model=model,
            tools=[search_policy],
            system_prompt="""당신은 청년 교육 정책 전문가입니다.
                교육 지원, 장학금, 교육 훈련 프로그램, 학비 지원, 등록금 지원 등에 대해 답변합니다.

                [중요 규칙]
                - 답변하기 전에 반드시 search_policy 도구를 먼저 호출하여 정책을 검색하세요.
                - 검색된 정책 내용에만 근거하여 답변하세요. 도구 결과에 없는 정책이나 수치를 임의로 만들어내지 마세요.
                - 검색 결과가 없으면, 관련 정책을 찾지 못했다고 솔직하게 안내하세요.

                정책 정보를 참고하여 친절하고 정확한 답변을 제공하세요.
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
                ✓ 신청 조건 및 제한사항 검토
                ✓ 소속 대학 담당자 문의

                [추가 문의사항]
                - 전화: 1588-1234 (평일 9-18시)
                - 이메일: support@company.com
                - 카카오톡: kakaotalk

                친절하게 안내해드리겠습니다.
                감사합니다."""
        )

    # 에이전트 실행 메소드
    async def run(self, question: str, user_profile: dict | None = None) -> DomainResult:
        token = _last_search_policies.set([])   # 이 task 컨텍스트만 초기화
        try:
            # 로그인 유저면 프로필을 함께 전달해 맞춤 답변 유도 (guest는 question만)
            content = question
            if user_profile:
                content = f"[사용자 정보]\n{user_profile}\n\n[질문]\n{question}"
            result = await self.agent.ainvoke(
                {"messages": [{"role": "user", "content": content}]}
            )
            policies = _last_search_policies.get()
        finally:
            _last_search_policies.reset(token)

        text = result["messages"][-1].content
        source = "rag" if policies else "none"
        return DomainResult(text=text, policies=policies, category=_CATEGORY, source=source)


EducationAgentDep = Annotated[EducationAgent, Depends(EducationAgent)]
