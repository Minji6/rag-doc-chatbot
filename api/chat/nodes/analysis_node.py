from langchain.chat_models import init_chat_model
from api.chat.state import ShareState

_llm = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0.0)

SYSTEM = """사용자 질문을 읽고 아래 도메인 중 하나만 출력하세요.
housing / employment / education / finance"""


async def analysis_node(state: ShareState) -> dict:
    result = await _llm.ainvoke([
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": state["user_inquiry"]},
    ])
    return {"inquiry_analysis": result.content.strip()}
