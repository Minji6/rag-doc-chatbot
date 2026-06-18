from ..state import ShareState
from ..constants import CATEGORY_ROUTING, DEFAULT_CATEGORY


def route_node_fun(state: ShareState) -> str:
    return CATEGORY_ROUTING.get(state["category"], CATEGORY_ROUTING[DEFAULT_CATEGORY])