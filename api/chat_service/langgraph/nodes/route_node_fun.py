from ..state import ShareState
from ..constants import CATEGORY_ROUTING


def route_node_fun(state: ShareState) -> list[str]:
    """analysis_node의 category 리스트를 도메인 검색 노드 이름 리스트로 변환.

    LangGraph는 conditional_edges가 list[str]을 반환하면 fan-out(병렬 실행)한다.
    분야를 특정하지 못하면(빈 리스트/미분류) **전 분야**를 fan-out한다.
    (과거에는 주거 한 분야로 fallback했으나, 의도가 모호할 때 한 분야만 보면
     사용자가 원한 정책을 놓칠 수 있어 4개 도메인을 모두 검색하도록 변경)

    단, 정책과 무관한 일반대화(is_general)면 검색을 건너뛰고 general 노드로 보낸다.
    """
    # 일반대화는 정책 검색 fan-out보다 우선 — 인사·잡담에 정책 검색이 도는 것을 차단.
    if state.get("is_general"):
        return ["general"]

    categories = state.get("category") or []
    routed: list[str] = []
    for c in categories:
        node = CATEGORY_ROUTING.get(c)
        if node and node not in routed:
            routed.append(node)
    if not routed:
        # 의도 불명 → 전 분야 검색 (composer가 결과 있는 도메인만 조립)
        return list(CATEGORY_ROUTING.values())
    return routed
