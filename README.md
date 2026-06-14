# 📚 Dev Doc RAG Chatbot

> FastAPI·LangChain·SQLAlchemy 공식 문서와 수업 자료를 RAG로 검색해 출처가 보장된 답변을 제공하는 개발자 학습 복습 AI 챗봇

---

## 1. About Project

| 항목 | 내용 |
|------|------|
| 프로젝트 목적 | 공식 문서와 수업 자료를 RAG로 검색해 버전 고정·출처 보장 답변을 제공하는 개발자 학습 코치 |
| 개발 기간 | 2주 (자세한 기간은 추후 추가 예정입니다)|
| 백엔드 레포 | [rag-doc-chatbot](https://github.com/Minji6/rag-doc-chatbot) |
| 프론트엔드 레포 | [rag-doc-chatbot-frontend](https://github.com/Minji6/rag-doc-chatbot-frontend) |

### 차별화

- 공식 문서(FastAPI·LangChain·SQLAlchemy) + 수업 자료를 별도 컬렉션으로 분리해 동시 검색
- 문서에 없는 내용은 명확하게 거절 (할루시네이션 방지)
- 인덱싱 시점 버전 고정으로 버전 보장
- 범용 AI와의 차별점: 공식 문서 출처 명시 + 수업 자료 연동 + Multi-Agent 구조

---

## 2. Team Members

<table>
  <tbody>
    <tr>
      <td align="center">
        <a href="https://github.com/Minji6">
          <img src="https://github.com/Minji6.png" width="100px;" alt="김민지"/>
          <br />
          <sub><b>김민지</b></sub>
        </a>
        <br />
        <sub>팀장 · FullStack</sub>
        <br />
        <a href="https://github.com/Minji6">GitHub</a>
      </td>
      <td align="center">
        <a href="https://github.com/dlwldP">
          <img src="https://github.com/dlwldP.png" width="100px;" alt="dlwldP"/>
          <br />
          <sub><b>이지예</b></sub>
        </a>
        <br />
        <sub>팀원 · FullStack</sub>
        <br />
        <a href="https://github.com/dlwldP">GitHub</a>
      </td>
      <td align="center">
        <a href="https://github.com/garden-kim-git">
          <img src="https://github.com/garden-kim-git.png" width="100px;" alt="garden-kim-git"/>
          <br />
          <sub><b>김정원</b></sub>
        </a>
        <br />
        <sub>팀원 · FullStack</sub>
        <br />
        <a href="https://github.com/garden-kim-git">GitHub</a>
      </td>
      <td align="center">
        <a href="https://github.com/qqqkyj">
          <img src="https://github.com/qqqkyj.png" width="100px;" alt="qqqkyj"/>
          <br />
          <sub><b>강연주</b></sub>
        </a>
        <br />
        <sub>팀원 · FullStack</sub>
        <br />
        <a href="https://github.com/qqqkyj">GitHub</a>
      </td>
    </tr>
  </tbody>
</table>

---

## 3. Key Features (수정 예정)

| # | 기능 | 설명 |
|---|------|------|
| 1 | 공식문서 + 수업자료 동시 검색 RAG 답변 | 두 컬렉션 동시 검색, 한국어 요약 + 원문 청크 + 출처 URL + 버전 구조화 출력 |
| 2 | 에러 로그 → 문서 + 수업 개념 연결 | 에러 로그 입력 시 원인 분석 + 해결책 + 수업 몇 주차 개념 연결 출력 |
| 3 | 라이브러리별 Multi-Agent 조율 | FastAPI·LangChain·SQLAlchemy 전담 Agent 분리, 조건 분기·병렬 처리 후 취합 |
| 4 | Doc Linter (코드 검증) | 코드 입력 시 deprecated 여부·권장 대체 패턴·출처 구조화 출력 |
| 5 | 대화 히스토리 유지 + 자동 요약 | PostgresSaver 영구 저장, 토큰 초과 시 SummarizationMiddleware 자동 요약 |
| 6 | 복습 퀴즈 자동 생성 | 문서 청크 기반 Few-Shot 퀴즈 생성, question·options·answer·explanation 구조화 출력 |
| 7 | 스트리밍 답변 출력 | astream + StreamingResponse 실시간 출력 |
| 8 | 공식문서 임베딩 관리 | URL fetch → 청킹 → PGVector 저장, 컬렉션 분리 관리 |

---

## 4. ERD

> 추후 추가 예정

---

## 5. Technology Stack

| 분류 | 기술 |
|------|------|
| Frontend | Next.js · React · JavaScript · Bootstrap 5 · Axios |
| Backend | Python · FastAPI · SQLAlchemy (async) |
| AI | LangChain · LangGraph · OpenAI |
| Database | PostgreSQL · pgvector |
| DB 관리 | pgAdmin |
| Infra | Docker |
| Tools | GitHub · Visual Studio Code |

---

## 6. Development Workflow

### 브랜치 전략

Git Flow를 기반으로 하되, 단기 프로젝트 규모에 맞게 `hotfix`·`release` 브랜치는 필요 시에만 사용합니다.

![branch strategy](./docs/git-flow.png)

| 브랜치 | 역할 | 규칙 |
|--------|------|------|
| `main` | 최종 배포본 | 직접 push 금지, dev에서만 머지 |
| `dev` | 통합 개발 | feat/* PR 리뷰 후 머지 |
| `feat/{기능명}` | 기능 단위 개발 | 완료 후 dev로 PR |

---

## 7. Convention

### 네이밍 컨벤션

함수·변수명은 언어 관례에 따르고, DB 컬럼명은 스네이크 케이스를 기본으로 한다.

```
// Python - 스네이크 케이스
get_chat_response, chat_history

// JavaScript - 카멜 케이스
sendMessage, chatHistory

// DB 컬럼 - 스네이크 케이스
created_at, user_id

// 클래스명 - 파스칼 케이스
ChatRequest, EmbeddingService
```

---

### 커밋 컨벤션

```
Feat: RAG 답변 API 구현

- 공식문서·수업자료 두 컬렉션 동시 검색 기능 추가
- Pydantic으로 출처 URL·버전·요약 구조화 출력
- 문서에 없는 질문 거절 처리 추가
```

| 타입 | 설명 |
|------|------|
| `Feat` | 새로운 기능 추가 |
| `Fix` | 버그 수정 |
| `Docs` | 문서 수정 |
| `Style` | 코드 formatting 등 코드 변경 없는 경우 |
| `Refactor` | 코드 리팩토링 |
| `Test` | 테스트 코드 추가 |
| `Chore` | 패키지 매니저, .gitignore 등 기타 수정 |
| `Rename` | 파일·폴더명 수정 및 이동 |
| `Remove` | 파일 삭제 |
| `!HOTFIX` | 치명적인 버그 긴급 수정 |
