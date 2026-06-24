from ..state import ShareState
from ..constants import CATEGORY_ROUTING, DEFAULT_CATEGORY


def route_node_fun(state: ShareState) -> list[str]:
    """analysis_node의 category 리스트를 도메인 검색 노드 이름 리스트로 변환.

    LangGraph는 conditional_edges가 list[str]을 반환하면 fan-out(병렬 실행)한다.
    빈 리스트 / 미분류 카테고리는 DEFAULT_CATEGORY로 fallback.
    """
    categories = state.get("category") or []
    routed: list[str] = []
    for c in categories:
        node = CATEGORY_ROUTING.get(c)
        if node and node not in routed:
            routed.append(node)
    if not routed:
        routed.append(CATEGORY_ROUTING[DEFAULT_CATEGORY])
    return routed
