from typing import Annotated, Literal, TypedDict

from langgraph.graph import add_messages

from api.chat_service.langgraph.model import UserProfile

# from api.welfare.model import UserProfile


class ShareState(TypedDict):
    """청년정책 챗봇 공유 상태.

    CLAUDE.md 10번 원칙에 따라 구조화된 정보는 str로 뭉치지 않고
    실제 타입(Pydantic 모델/리터럴 등)으로 선언한다.

    이 파일은 슈퍼바이저(analysis_node, 라우팅)가 최종적으로 관리하지만,
    도메인 에이전트(WelfareAgent 등)가 의존하는 인터페이스이므로
    본 작업에서 합의용으로 함께 정의해둔다.
    """

    # 대화 메시지 (필수)
    messages: Annotated[list, add_messages]

    # 사용자 원본 질문
    user_query: str

    # 역할 - guest(기본값) 또는 user. user일 때만 user_profile이 채워짐.
    user_role: Literal["guest", "user"]

    # user인 경우 DB에서 조회된 프로필. guest는 None.
    user_profile: UserProfile | None

    # 슈퍼바이저(analysis_node)가 분류한 분야 카테고리 ("복지"/"주거"/"취업"/"교육" 등)
    category: str

    # 도메인 에이전트가 채우는 최종 응답
    final_response: str