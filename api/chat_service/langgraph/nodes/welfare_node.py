import logging
from ..state import ShareState, DomainResult
from ..constants import agent_mode
from ..agents.welfare_agent import WelfareAgent, _CATEGORY
from ..tools.welfare_eligibility import build_eligibility_report

logger = logging.getLogger(__name__)
agent = WelfareAgent()


async def welfare_node(state: ShareState) -> dict:
    """복지문화 정책 생성 노드.

    검색 노드(welfare_search_node)가 state.domain_knowledge/policies["welfare"]에 넣어둔 값을 읽어
    WelfareAgent에 전달하고, DomainResult를 받아 state.domain_results["welfare"]에 기록한다.

    상세조회 + 로그인 시에는 복지 특화 자격 검증(build_eligibility_report)을 정책에 첨부한다.
    프론트 상세 모달의 '자격 검증' 섹션이 policy.eligibility({status,items,summary})를 렌더한다.
    """
    logger.info("복지문화 정책 생성 노드 실행")

    inquiry = state["user_inquiry"]
    image_context = state.get("image_context", "")
    if image_context:
        inquiry = f"{inquiry}\n\n[첨부 이미지 내용]\n{image_context}"

    inquiry_types = state.get("inquiry_type", [])
    inquiry_type = agent_mode(inquiry_types)

    text, policies, source, merged_profile, suggestions = await agent.run(
        inquiry=inquiry,
        knowledge=state.get("domain_knowledge", {}).get("welfare", ""),
        policies=state.get("domain_policies", {}).get("welfare", []),
        user_role=state.get("user_role", "guest"),
        user_profile=state.get("user_profile"),
        inquiry_type=inquiry_type,
        messages=state.get("messages", []),
    )
    result = DomainResult(text=text, policies=policies, category=_CATEGORY, source=source)
    out = {
        "domain_results": {"welfare": result},
        "user_profile": merged_profile,
        "suggestions": suggestions,
    }

    # 상세조회 포함 + 로그인 시, 복지 정책에 자격 검증 결과(구조화)를 첨부 (게스트=merged_profile 비어 스킵).
    # inquiry_type은 리스트라 "상세조회"가 다른 의도(추천/비교)와 섞여 올 수 있으므로 멤버십으로 판정한다.
    # (agent_mode는 우선순위로 하나로 축약해 상세조회가 가려질 수 있음)
    # - policies: 이번 턴 검색 정책 (일반 상세조회 경로)
    # - resolved_policies: 직전 정책 후속질문 경로. supervisor가 최종 policies를 resolved로
    #   덮어쓰므로, 여기서 붙여야 프론트 모달의 '자격 검증'이 누락되지 않는다.
    if "상세조회" in inquiry_types and merged_profile:
        for policy in policies:
            report = build_eligibility_report(merged_profile, policy)
            if report:
                policy["eligibility"] = report

        resolved = state.get("resolved_policies") or []
        touched = False
        for policy in resolved:
            if policy.get("category") == _CATEGORY:
                report = build_eligibility_report(merged_profile, policy)
                if report:
                    policy["eligibility"] = report
                    touched = True
        if touched:
            out["resolved_policies"] = resolved

    return out
