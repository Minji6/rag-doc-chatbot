from typing import Any
from langchain.tools import tool

@tool
def search_review(query: str) -> Any:
    raise NotImplementedError
