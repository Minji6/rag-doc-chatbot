# 📚 청년정책지원 챗봇 — Backend

온통청년 API 정책 데이터를 RAG로 검색해 주거·취업·교육·금융 분야별 맞춤 답변을 제공하는 청년정책 AI 챗봇 백엔드

---

## 1. About Project

| 항목 | 내용 |
|------|------|
| 프로젝트 목적 | 청년 정책 정보를 RAG로 검색해 분야별 맞춤 답변을 제공하는 청년정책 지원 챗봇 |
| 프론트엔드 레포 | [rag-doc-chatbot-frontend](https://github.com/Minji6/rag-doc-chatbot-frontend) |
| 개발 기간 | 2026.06.16 ~ 2026.06.30 (2주) |

---

## 2. Team Members

| 역할 | 이름 |
|------|------|
| 팀장 · FullStack | 김민지 |
| 팀원 · FullStack | 강연주 |
| 팀원 · FullStack | 김정원 |
| 팀원 · FullStack | 이지예 |

---

## 3. Key Features

| # | 기능 | 설명 | 활용 챕터 |
|---|------|------|----------|
| 1 | 도메인별 정책 RAG 답변 | 주거·취업·교육·금융 전담 에이전트가 PGVector에서 분야별 정책 검색 후 한국어 답변 생성 | sec07, sec08 |
| 2 | Multi-Agent Supervisor 라우팅 | 사용자 의도 분석 후 4개 도메인 에이전트 중 하나로 조건 분기 | sec09 |
| 3 | 대화 히스토리 유지 | 게스트는 InMemorySaver, 회원은 AsyncPostgresSaver로 영구 저장 | sec06 |
| 4 | 정책 데이터 수집 + 임베딩 | 온통청년 오픈 API → httpx 수집 → 청킹 → PGVector 저장 | sec08 |
| 5 | 회원가입 · 로그인 (JWT) | 회원/게스트 판별, AsyncPostgresSaver 전환 기준 | sec06 |
| 6 | 정책 상세 조회 + D-day | 정책 마감일 기반 D-day 계산, return_direct 도구로 즉시 반환 | sec07 |
| 7 | 스트리밍 답변 출력 | astream + StreamingResponse 실시간 출력 | sec02, sec05 |
| 8 | 맞춤 정책 추천 | 사용자 프로필 기반 similarity search로 상위 정책 추천 | sec04, sec08 |

---

## 4. ERD

> 추후 추가 예정

---

## 5. Technology Stack

| 분류 | 기술 |
|------|------|
| Backend | Python · FastAPI · SQLAlchemy (async) |
| AI | LangChain · LangGraph · OpenAI |
| Database | PostgreSQL · pgvector |
| DB 관리 | pgAdmin |
| Infra | Docker |
| Tools | GitHub · Visual Studio Code |

---

## 6. Getting Started

### 사전 요구사항
- Python 3.11+
- PostgreSQL + pgvector 확장
- Docker (DB 실행용)

### 설치 및 실행

```bash
# 1. 레포 클론
git clone https://github.com/Minji6/rag-doc-chatbot
cd rag-doc-chatbot

# 2. 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 패키지 설치
pip install -r requirements.txt

# 4. 환경변수 설정
cp .env.example .env
# .env 파일을 열어 값 입력

# 5. 실행
python main.py
```

### 환경변수 (.env)

```
OPENAI_API_KEY=
TAVILY_API_KEY=
DATABASE_URL=postgresql://postgres:비밀번호@localhost:5432/DB이름
DATABASE_URL_SQLALCHEMY=postgresql+psycopg://postgres:비밀번호@localhost:5432/DB이름
```

---

## 7. Development Workflow

### 브랜치 전략

| 브랜치 | 역할 | 규칙 |
|--------|------|------|
| `main` | 최종 배포본 | 직접 push 금지, dev에서만 머지 |
| `dev` | 통합 개발 | feat/* PR 리뷰 후 머지 |
| `feat/{기능명}` | 기능 단위 개발 | 완료 후 dev로 PR |

### PR 규칙
- `feat/*` → `dev` PR 생성 후 팀원 2명 이상 리뷰 후 머지
- PR 제목 형식: `[feat] RAG 답변 API 구현`
- `dev` → `main`은 전체 기능 완료 후 최종 1회 머지

---

## 8. Convention

### 커밋 컨벤션

| 타입 | 설명 |
|------|------|
| `Feat` | 새로운 기능 추가 |
| `Fix` | 버그 수정 |
| `Docs` | 문서 수정 |
| `Style` | 코드 formatting, 세미콜론 누락 등 코드 변경 없는 경우 |
| `Refactor` | 코드 리팩토링 |
| `Test` | 테스트 코드 추가 및 리팩토링 |
| `Chore` | 패키지 매니저 수정, .gitignore 등 기타 수정 |
| `Comment` | 필요한 주석 추가 및 변경 |
| `Rename` | 파일 또는 폴더 명 수정 및 이동 |
| `Remove` | 파일 삭제 |
| `!HOTFIX` | 급하게 치명적인 버그를 고쳐야 하는 경우 |

**커밋 메시지 규칙**
- 커밋 유형은 영어 대문자로 작성
- 제목과 본문은 빈 행으로 분리
- 제목 첫 글자 대문자, 끝에 `.` 금지
- 제목은 영문 기준 50자 이내
- 본문에는 무엇을·왜 변경했는지 설명 (어떻게 X)
- 여러 항목은 글머리 기호로 작성

```
Feat: RAG 답변 API 구현

- 공식문서·수업자료 두 컬렉션 동시 검색 기능 추가
- Pydantic으로 출처 URL·버전·요약 구조화 출력
- 문서에 없는 질문 거절 처리 추가
```

### 코드 컨벤션 (Python / FastAPI)

| 대상 | 규칙 | 예시 |
|------|------|------|
| 클래스명 | PascalCase | `ChatRequest`, `EmbeddingService` |
| 함수·변수명 | snake_case | `get_chat_response`, `chat_history` |
| 상수 | UPPER_SNAKE_CASE | `MAX_TOKENS`, `COLLECTION_NAME` |
| 파일명 | snake_case | `chat_controller.py`, `rag_service.py` |
| 라우터 prefix | kebab-case | `/api/chat`, `/api/doc-linter` |
| DB 컬럼명 | snake_case | `created_at`, `board_id` |

**추가 규칙**
- 함수 하나는 하나의 역할만 수행
- controller는 요청/응답만, 비즈니스 로직은 service로 분리
- 비동기 함수는 `async def` 사용
- 타입 힌트 필수 작성
