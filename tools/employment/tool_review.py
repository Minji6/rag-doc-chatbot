from typing import Any
from langchain_core.tools import tool

@tool
def search_review(query: str) -> Any:
    raise NotImplementedError
