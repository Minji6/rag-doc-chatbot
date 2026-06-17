from typing import Annotated, TypedDict
from langgraph.graph import add_messages


class ShareState(TypedDict):
    messages: Annotated[list, add_messages]
    user_inquiry: str
    user_id: str | None
    conversation_id: str
    inquiry_analysis: str
    final_response: str
