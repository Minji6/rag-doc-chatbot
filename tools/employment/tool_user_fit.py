from typing import Any
from langchain.tools import tool

@tool
def compare_user_fit(
    user_profile: dict[str, Any],
    item: dict[str, Any],
) -> Any:
    raise NotImplementedError
