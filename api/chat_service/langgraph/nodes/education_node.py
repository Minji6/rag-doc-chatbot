import logging
import re
from ..state import ShareState, DomainResult
from ..agents.education_agent import EducationAgent
from ..constants import AGENT_CATEGORY, agent_mode

logger = logging.getLogger(__name__)
agent = EducationAgent()

_CATEGORY = AGENT_CATEGORY["education"]
_POLICY_HEADER_RE = re.compile(r"^### (.+)$", re.MULTILINE)


async def education_node(state: ShareState) -> dict:
    """교육 정책 생성 노드.

    검색 노드(education_search_node)가 state.domain_knowledge/policies["education"]에 넣어둔
    값을 읽어 답변을 생성하고 DomainResult로 반환한다. (검색은 하지 않음)
    """
    logger.info("교육 정책 생성 노드 실행")

    knowledge = state.get("domain_knowledge", {}).get("education", "")
    policies = state.get("domain_policies", {}).get("education", [])

    inquiry = state["user_inquiry"]
    image_context = state.get("image_context", "")
    if image_context:
        inquiry = f"{inquiry}\n\n[첨부 이미지 내용]\n{image_context}"

    resolved_mode = agent_mode(state.get("inquiry_type", []))
    text, suggestions = await agent.run(
        inquiry=inquiry,
        knowledge=knowledge,
        policies=policies,
        user_profile=state.get("user_profile"),
        inquiry_type=resolved_mode,
    )

    # 상세조회는 정책 1개에 집중하는 모드다. 검색 단계는 유사 정책까지 대비해 k=2로 가져오지만,
    # 카드 UI(응답 policies)는 에이전트가 실제로 "### 정책명"으로 설명한 정책만 보여준다.
    # 그렇지 않으면 질문과 무관한 2번째 검색 결과가 그대로 카드로 함께 노출된다.
    if resolved_mode == "상세조회" and policies:
        displayed = set(_POLICY_HEADER_RE.findall(text))
        filtered = [p for p in policies if p.get("plcyNm") in displayed]
        if filtered:
            policies = filtered

    source = "rag" if policies else "none"
    result = DomainResult(
        text=text, policies=policies, category=_CATEGORY, source=source
    )
    return {"domain_results": {"education": result}, "suggestions": suggestions}
