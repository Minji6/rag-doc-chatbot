# 프로젝트: 청년정책지원 챗봇 Backend

## 기술 스택
- FastAPI 0.116 (Python 3.11+, async)
- LangChain 1.2 / LangGraph 1.0, OpenAI(gpt-4o-mini), Tavily
- PostgreSQL + pgvector, SQLAlchemy(async), psycopg
- 테스트: pytest / 타입체크: mypy / 포맷: autopep8

## 아키텍처 규칙
- CRITICAL: 도메인 에이전트(`agents/{domain}_agent.py`)는 응답 **fragment만** 생성한다. 인사말·전체 응답 헤더 조립은 `composer` 노드가 단독으로 책임진다. 인사는 대화의 첫 턴에서만 붙인다.
- CRITICAL: `supervisor.py`의 그래프 위상(`analysis → fan-out({domain}_search → {domain}) → composer`)을 변경하는 리팩토링은 반드시 `docs/ADR.md`에 새 ADR을 추가한 뒤 진행한다.
- CRITICAL: controller는 요청/응답 매핑만 하고, 비즈니스 로직은 service/langgraph 계층에 둔다.
- CRITICAL: 정책 메타데이터 화이트리스트(예: `POLICY_METADATA_FIELDS`)에 필드를 추가할 때는 온통청년 API 원본 필드명과 대소문자까지 정확히 일치하는지 검증한다. 불일치 시 값이 조용히 `None`이 되어 자격 판정이 항상 통과되는 버그로 이어진다.
- 비동기 I/O 함수는 `async def`, 타입 힌트 필수.
- 함수 하나는 하나의 역할만 수행한다(단일 책임).

## 개발 프로세스
- CRITICAL: 이 하네스는 **리팩토링 전용**이다. 새 기능·새 엔드포인트 추가는 scope 밖이며, 요청받지 않는 한 만들지 않는다.
- CRITICAL: 각 phase는 기존 동작(API 응답, DB 스키마)을 바꾸지 않는다. 동작 변경이 필요하면 별도로 사용자에게 알리고 phase를 분리한다.
- 기존 테스트가 있다면 절대 깨뜨리지 않는다. 테스트가 없는 모듈을 리팩토링할 경우, 리팩토링 전 동작을 고정하는 최소 테스트를 먼저 작성한다(가능한 범위 내에서 TDD).
- 커밋 메시지는 이 프로젝트의 컨벤션을 따른다: `Feat:`, `Fix:`, `Docs:`, `Style:`, `Refactor:`, `Test:`, `Chore:`, `Comment:`, `Rename:`, `Remove:`, `!HOTFIX:` — 대문자 시작, 제목 끝에 마침표 금지, 영문 기준 50자 이내.

## 코드 컨벤션 (Python / FastAPI)
| 대상 | 규칙 | 예시 |
|------|------|------|
| 클래스명 | PascalCase | `ChatRequest` |
| 함수·변수명 | snake_case | `get_chat_response` |
| 상수 | UPPER_SNAKE_CASE | `MAX_TOKENS` |
| 파일명 | snake_case | `chat_controller.py` |
| 라우터 prefix | kebab-case | `/api/chat` |
| DB 컬럼명 | snake_case | `created_at` |

## 명령어
```
python main.py                                  # 서버 실행
python -m pytest                                 # 테스트
python -m mypy .                                  # 타입체크
python -m autopep8 --in-place --recursive api/    # 포맷
```
