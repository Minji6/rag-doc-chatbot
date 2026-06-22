from typing import Any
from langchain_core.tools import tool

@tool
def compare_user_fit(
    user_profile: dict[str, Any],
    item: dict[str, Any],
) -> Any:
    raise NotImplementedError
