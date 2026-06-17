from langchain.chat_models import init_chat_model
from api.chat.state import ShareState

_llm = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0.0)

SYSTEM = "당신은 청년 교육 정책 전문가입니다. 검색된 정책 정보를 바탕으로 친절하게 답변하세요."


async def education_node(state: ShareState) -> dict:
    # TODO: PGVector 검색 결과를 context에 추가
    result = await _llm.ainvoke([
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": state["user_inquiry"]},
    ])
    return {"final_response": result.content}
