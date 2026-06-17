from api.chat.state import ShareState

VALID = {"housing", "employment", "education", "finance"}


def route_node_fun(state: ShareState) -> str:
    domain = state.get("inquiry_analysis", "")
    return domain if domain in VALID else "finance"
