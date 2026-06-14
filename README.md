# 📚 Dev Doc RAG Chatbot — Backend

FastAPI·LangChain·SQLAlchemy 공식 문서와 수업 자료를 RAG로 검색해 출처가 보장된 답변을 제공하는 개발자 학습 복습 AI 챗봇 백엔드

---

## 1. About Project

| 항목 | 내용 |
|------|------|
| 프로젝트 목적 | 공식 문서와 수업 자료를 RAG로 검색해 버전 고정·출처 보장 답변을 제공하는 개발자 학습 코치 |
| 프론트엔드 레포 | [rag-doc-chatbot-frontend](https://github.com/Minji6/rag-doc-chatbot-frontend) |
| 개발 기간 | 2주 |

### 특이사항
- 공식 문서(FastAPI·LangChain·SQLAlchemy) + 수업 자료를 별도 컬렉션으로 분리해 동시 검색
- 문서에 없는 내용은 명확하게 거절 (할루시네이션 방지)
- 인덱싱 시점 버전 고정으로 버전 보장

---

## 2. Team Members

| 역할 | 이름 |
|------|------|
| 팀장 · FullStack | 김민지 |
| 팀원 · FullStack | - |
| 팀원 · FullStack | - |
| 팀원 · FullStack | - |

---

## 3. Key Features

| # | 기능 | 설명 | 활용 챕터 |
|---|------|------|-----------|
| 1 | 공식문서 + 수업자료 동시 검색 RAG 답변 | 두 컬렉션 동시 검색, 한국어 요약 + 원문 청크 + 출처 URL + 버전 구조화 출력 | sec04, sec07, sec08 |
| 2 | 에러 로그 → 문서 + 수업 개념 연결 | 에러 로그 입력 시 원인 분석 + 해결책 + 수업 개념 연결 출력 | sec03, sec07, sec08 |
| 3 | 라이브러리별 Multi-Agent 조율 | FastAPI·LangChain·SQLAlchemy 전담 Agent 분리, 조건 분기·병렬 처리 후 취합 | sec07, sec08, sec09 |
| 4 | Doc Linter (코드 검증) | deprecated 여부·권장 대체 패턴·출처 구조화 출력 | sec04, sec07, sec08 |
| 5 | 대화 히스토리 유지 + 자동 요약 | PostgresSaver 영구 저장, 토큰 초과 시 자동 요약 압축 | sec06 |
| 6 | 복습 퀴즈 자동 생성 | 문서 청크 기반 Few-Shot 퀴즈 생성, Pydantic 구조화 출력 | sec03, sec04 |
| 7 | 스트리밍 답변 출력 | astream + StreamingResponse 실시간 출력 | sec02, sec05 |
| 8 | 공식문서 임베딩 관리 | URL fetch → 청킹 → PGVector 저장, 컬렉션 분리 관리 | sec08 |

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
