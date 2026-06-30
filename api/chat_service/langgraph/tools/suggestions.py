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
import re

_SEPARATOR = "---SUGGESTIONS---"
_SEPARATOR_ALT = "SUGGESTIONS---"  # LLM이 앞 --- 를 생략하는 경우 방어

# raw_decode는 주어진 위치에서 JSON 값 하나만 파싱하고 '뒤에 붙은 텍스트는 무시'한다.
# → LLM이 배열 뒤에 사족(설명)을 덧붙여도 배열만 깔끔히 추출할 수 있다.
_DECODER = json.JSONDecoder()

# 구분자 없이 배열만 노출됐을 때, 배열 바로 앞 머리말("다음에 할 법한 질문" 등)을 함께 떼어내기 위한 패턴.
_SUGGESTION_HEADER = re.compile(r"(질문|추천|제안|follow)", re.IGNORECASE)

# 답변 프롬프트 끝에 덧붙여, LLM이 본문 뒤에 follow-up 질문 JSON 배열을 출력하도록 지시한다.
SUGGESTIONS_PROMPT = (
    "\n\n답변 작성 후 반드시 아래 구분자와 함께 사용자가 다음에 할 법한 "
    "follow-up 질문 3개를 JSON 배열로 추가하세요.\n"
    f"{_SEPARATOR}\n"
    '["질문1", "질문2", "질문3"]'
)


def _decode_str_list_at(segment: str, bracket_pos: int) -> list[str] | None:
    """segment의 bracket_pos('[' 위치)에서 JSON 배열을 파싱해 문자열 리스트로 반환.

    배열이 아니거나 파싱 실패면 None. 배열 뒤의 잔여 텍스트는 raw_decode가 무시한다.
    """
    try:
        value, _ = _DECODER.raw_decode(segment, bracket_pos)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, list):
        return None
    return [s for s in value if isinstance(s, str)]


def parse_suggestions(content: str) -> tuple[str, list[str]]:
    """LLM 응답에서 본문과 suggestions를 분리한다.

    1순위: 지시한 구분자(---SUGGESTIONS---) 기준 분리. 구분자 뒤에 사족이 붙어도
           배열만 추출한다(raw_decode).
    2순위(방어): 구분자를 빠뜨리고 "다음에 할 법한 질문" 같은 머리말과 함께 배열을
           본문 끝에 붙이는 경우 — 맨 끝 문자열 배열을 떼어내 본문에서 제거한다.
           (배열이 본문에 raw로 노출되던 버그 방지)

    Args:
        content: LLM 응답 전문 (본문 + ---SUGGESTIONS--- + JSON 배열)
    Returns:
        tuple[str, list[str]]: (본문 텍스트, follow-up 질문 목록)
    """
    # 정상 구분자(---SUGGESTIONS---) 또는 앞 --- 생략 변형(SUGGESTIONS---) 모두 처리.
    sep = _SEPARATOR if _SEPARATOR in content else (_SEPARATOR_ALT if _SEPARATOR_ALT in content else None)
    if sep:
        body, _, tail = content.partition(sep)
        bracket = tail.find("[")
        if bracket != -1:
            items = _decode_str_list_at(tail, bracket)
            if items:
                return body.strip(), items
        # 구분자는 있지만 JSON 배열이 없거나 파싱 실패 → body만 반환 (구분자 이후 텍스트 제거).
        return body.strip(), []

    # 구분자 누락 방어 — 맨 끝에서 닫히는 문자열 배열만 후보로 본다(본문 중간 대괄호 오인 방지).
    stripped = content.rstrip()
    if not stripped.endswith("]"):
        return content.strip(), []
    bracket = stripped.rfind("[")
    while bracket != -1:
        items = _decode_str_list_at(stripped, bracket)
        if items:
            head = content[:bracket].rstrip()
            nl = head.rfind("\n")
            last_line = head[nl + 1:]
            # 배열 바로 앞 한 줄이 짧은 머리말이면 함께 제거
            cut = (nl + 1 if nl != -1 else 0) if (len(last_line) <= 40 and _SUGGESTION_HEADER.search(last_line)) else bracket
            return content[:cut].rstrip(), items
        bracket = stripped.rfind("[", 0, bracket)
    return content.strip(), []
