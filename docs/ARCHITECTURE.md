# ARCHITECTURE — 청년정책지원 챗봇 Backend

## 기술 스택

- **Backend**: Python 3.11+, FastAPI 0.116, SQLAlchemy(async), psycopg
- **AI**: LangChain 1.2, LangGraph 1.0, OpenAI(gpt-4o-mini), Tavily(웹 검색 보완)
- **Database**: PostgreSQL + pgvector
- **Lint/Test**: mypy, autopep8, pytest

## 디렉터리 구조

```
api/
├── chat_service/
│   ├── controller.py
│   ├── policy_memory.py      # 후속질문 해소용 직전 턴 정책 메모리
│   └── langgraph/
│       ├── supervisor.py     # StateGraph 정의 (ChatbotSupervisor) — fan-out → fan-in
│       ├── state.py          # ShareState (그래프 상태 스키마)
│       ├── constants.py
│       ├── nodes/            # analysis, contextualize, image_analysis,
│       │                     # {domain}_search, {domain}, composer, general, route_node_fun
│       ├── agents/           # housing / employment / education / welfare 에이전트
│       └── tools/            # age, dday, eligibility, welfare_eligibility,
│                              # policy_search, suggestions
├── auth_service/             # controller / model / service 3계층
├── history_service/          # 대화 기록 (InMemorySaver / AsyncPostgresSaver)
├── upload_service/           # 정책 데이터 수집 · 임베딩
├── models/
└── common/                   # 예외 처리, DB 커넥션 풀 등 공통 유틸
```

## 데이터 흐름 (LangGraph)

```
image_analysis → contextualize → analysis
                                    │
                    route_node_fun (조건부 fan-out)
                                    │
        ┌──────────┬──────────┬──────────┬──────────┐
   housing_search employment_search education_search welfare_search   general
        │              │              │              │                │
    housing_node  employment_node education_node  welfare_node        │
        └──────────┴──────────┴──────────┴──────────┴────────────────┘
                                    │
                                composer (fan-in)
                                    │
                                   END
```

- `analysis` 노드가 사용자 발화의 도메인 의도를 분석하고 `route_node_fun`이 조건부로 fan-out 대상을 결정한다.
- 각 도메인은 `{domain}_search_node`(정책 검색) → `{domain}_node`(에이전트가 fragment 생성)의 2단계로 구성된다.
- 도메인 에이전트는 **fragment만** 생성한다. 인사말·헤더 등 응답의 뼈대는 `composer`가 단독으로 책임진다(인사는 대화 첫 턴에서만).
- `composer`가 여러 도메인의 fragment를 모아 최종 응답으로 조립(fan-in)한다.

## 핵심 설계 규칙

- **컨트롤러-서비스 분리**: controller는 요청/응답만 처리하고 비즈니스 로직은 service/langgraph 계층에 둔다.
- **비동기 우선**: 모든 I/O 바운드 함수는 `async def`.
- **타입 힌트 필수**.
- **정책 메타데이터 화이트리스트**: `POLICY_METADATA_FIELDS`(또는 동급 화이트리스트 상수)에 필드를 추가할 때는 온통청년 API 원본 필드명(대소문자 포함)과 정확히 일치하는지 확인한다. 필드명이 존재하지 않거나 대소문자가 다르면 값이 항상 `None`으로 취급되어 자격 판정(소득·학력 등)이 조용히 항상 통과되는 버그로 이어질 수 있다.

## 리팩토링 시 지켜야 할 경계

- `supervisor.py`의 그래프 위상(fan-out → fan-in 구조)은 변경하지 않는다. 노드 내부 구현 개선은 가능하나 그래프 엣지 구조 변경은 별도 ADR 없이는 하지 않는다.
- `nodes/`와 `agents/`의 책임 분리(검색 vs 응답 생성)를 유지한다.
