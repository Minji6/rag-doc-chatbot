"""follow-up 질문(suggestions) 생성 공통 유틸.

지금까지 education_agent에만 있던 '답변 끝에 ---SUGGESTIONS--- 구분자로 follow-up
질문 3개를 붙이고 파싱'하는 로직을 분리해, 어느 도메인 에이전트든 동일하게
suggestions를 생성할 수 있게 한다.

사용법:
    from ..tools import SUGGESTIONS_PROMPT, parse_suggestions

    prompt += SUGGESTIONS_PROMPT
    result = await agent.ainvoke(...)
    text, suggestions = parse_suggestions(result["messages"][-1].content)
"""
import json

_SEPARATOR = "---SUGGESTIONS---"

# 답변 프롬프트 끝에 덧붙여, LLM이 본문 뒤에 follow-up 질문 JSON 배열을 출력하도록 지시한다.
SUGGESTIONS_PROMPT = (
    "\n\n답변 작성 후 반드시 아래 구분자와 함께 사용자가 다음에 할 법한 "
    "follow-up 질문 3개를 JSON 배열로 추가하세요.\n"
    f"{_SEPARATOR}\n"
    '["질문1", "질문2", "질문3"]'
)


def parse_suggestions(content: str) -> tuple[str, list[str]]:
    """LLM 응답에서 본문과 suggestions를 분리한다.

    구분자가 없으면 전체를 본문으로 보고 빈 목록을 반환한다.

    Args:
        content: LLM 응답 전문 (본문 + ---SUGGESTIONS--- + JSON 배열)
    Returns:
        tuple[str, list[str]]: (본문 텍스트, follow-up 질문 목록)
    """
    if _SEPARATOR not in content:
        return content.strip(), []
    parts = content.split(_SEPARATOR, 1)
    text = parts[0].strip()
    try:
        suggestions = json.loads(parts[1].strip())
        if not isinstance(suggestions, list):
            suggestions = []
        else:
            suggestions = [s for s in suggestions if isinstance(s, str)]
    except (json.JSONDecodeError, IndexError):
        suggestions = []
    return text, suggestions
