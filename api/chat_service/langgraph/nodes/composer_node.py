# api/chat_service/langgraph/nodes/composer_node.py
"""Composer 노드 — 도메인 에이전트가 만든 본문 조각(fragment)을 하나의 답변으로 조립한다.

설계 (LangGraph supervisor-worker의 fan-in/aggregator 단계):
- 도메인 에이전트 4종은 "완성된 답변"이 아니라 "### 정책명 본문 조각"만 만든다.
- 이 노드가 cross-cutting 책임(도입부·분야 헤더·맥락 멘트·맺음말)을 **단독으로** 진다.
  → 인사 N번 / 헤더 중복·누락 / 분야 간 형식 비일관 문제를 원천 차단.

본문(### 정책 블록 / 비교 표)은 결정적으로 조립한다(변동성·토큰 0, 정확성 보장).
도입부·맥락 멘트(추천 이유·비교 관점)·맺음말만 LLM이 질문 맥락에 맞춰 생성한다.
LLM 실패 시 결정적 템플릿으로 폴백해 답변은 항상 나온다.
"""
import json
import logging
import re
from typing import Annotated, cast

from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

from ..state import DomainResult, ShareState
from ._policy_table import build_comparison_table, policy_key

logger = logging.getLogger(__name__)

_FALLBACK_MESSAGE = "죄송합니다, 답변을 생성하지 못했습니다. 다시 질문해주세요."

# 첫 대화에서만 1회 붙는 인사 템플릿. 후속 턴(대화기록 존재)에는 인사 없이 본문만.
# (대화가 이어지는 중에 매 턴 인사하면 어색하므로 conversation-start에만 부착)
_GREETING_TEMPLATE = "안녕하세요! 청년정책 안내를 도와드리는 챗봇입니다. 😊"

# 멀티 분야 답변을 항상 같은 순서로 정렬하기 위한 분야 출력 순서.
_CATEGORY_ORDER = ("주거", "일자리", "교육", "복지문화")

# 도메인 fragment에 인사/팀헤더/맺음말/분야헤더가 새어들어왔을 때 제거하는 방어 패턴.
# 1차 방어는 각 에이전트 프롬프트(OUTPUT_FORMAT_GUIDE)이고, 이 strip은 안전망이다.
_STRIP_PATTERNS = (
    re.compile(r"^[ \t]*#{1,3}[ \t]*(주거|일자리|교육|복지문화)[ \t]*$", re.MULTILINE),  # 분야 헤더
    re.compile(r"^.*정책팀[^\n]*[:：][ \t]*$", re.MULTILINE),                              # "🏠 주거정책팀 답변:"
    re.compile(r"^[ \t]*안녕하세요[^\n]*$", re.MULTILINE),                                  # 인사말
    re.compile(r"^[ \t]*감사합니다[.!]*[ \t]*$", re.MULTILINE),                             # 맺음말
    re.compile(r"^[ \t]*친절하게[^\n]*$", re.MULTILINE),                                    # "친절하게 안내..."
)


def _strip_boilerplate(text: str) -> str:
    """도메인 fragment에서 인사/팀헤더/맺음말/분야헤더를 제거한다 (안전망)."""
    for pattern in _STRIP_PATTERNS:
        text = pattern.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)  # 제거로 생긴 과도한 빈 줄 정리
    return text.strip()


def _collect_active_fragments(state: ShareState) -> list[DomainResult]:
    """텍스트가 있고 source != "none"인 도메인 결과만 분야 순서대로 정렬해 반환."""
    domain_results = state.get("domain_results") or {}
    active = [
        r for r in domain_results.values()
        if r.get("text") and r.get("source") != "none"
    ]
    order = {c: i for i, c in enumerate(_CATEGORY_ORDER)}
    active.sort(key=lambda r: order.get(r.get("category", ""), len(order)))
    return active


def _build_intro(categories: list[str]) -> str:
    """멀티 분야 도입부 1줄 (결정적 템플릿)."""
    return f"{', '.join(categories)} 분야의 정책을 안내해 드릴게요."


def _clean_comment_line(line: str) -> bool:
    """비교 코멘트로 남길 줄인지 판정 (표/정책블록 잔재 제거용 안전망).

    제거 대상: 마크다운 표 줄(| ... |), 헤딩 줄(#, ##, ### 정책 블록 헤더).
    → 에이전트가 비교 코멘트 대신 정책 블록을 흘려도 산문 코멘트만 남긴다.
    """
    stripped = line.lstrip()
    if stripped.startswith("|"):
        return False
    if stripped.startswith("#"):
        return False
    return True


def _build_domain_comments(fragments: list[DomainResult]) -> str:
    """비교 모드 — 각 도메인 에이전트가 만든 정성 코멘트를 분야별로 모은다 (하이브리드).

    표(정형 데이터)는 composer가, 코멘트(분야 판단)는 에이전트가 책임진다.
    혹시 에이전트가 표·정책블록(### )을 흘렸어도 안전망으로 제거하고 산문만 남긴다.
    """
    lines: list[str] = []
    for frag in fragments:
        comment = _strip_boilerplate(frag.get("text", ""))
        # 표 줄(| ... |)·헤딩(### 정책 블록)이 새어들어왔으면 제거 — 표는 위에서 이미 결정적으로 그렸음.
        comment = "\n".join(
            ln for ln in comment.splitlines() if _clean_comment_line(ln)
        ).strip()
        comment = re.sub(r"\n{2,}", " ", comment).strip()  # 블록 잔재로 생긴 줄바꿈 정리
        if comment:
            lines.append(f"- **{frag.get('category', '')}**: {comment}")
    if not lines:
        return ""
    return "**분야별 코멘트**\n" + "\n".join(lines)


def _collect_policies_for_compare(fragments: list[DomainResult]) -> list[dict]:
    """활성 도메인 결과에서 raw 정책 메타를 분야 순서대로 평탄화한다.

    각 정책에 분야 라벨을 보강한다(메타에 category가 없을 때 도메인 결과 category로 채움).
    """
    policies: list[dict] = []
    for frag in fragments:
        domain_category = frag.get("category", "")
        for policy in frag.get("policies", []):
            if not policy:
                continue
            if not policy.get("category"):
                policy = {**policy, "category": domain_category}
            policies.append(policy)
    return policies


# ── 맥락 멘트(LLM) ───────────────────────────────────────────────────────────
# 본문은 결정적으로 두고, 답변에 필요한 서술(도입·추천이유·비교관점·맺음말)만 LLM이 생성한다.

class _Commentary(BaseModel):
    """Composer가 본문 앞뒤에 붙일 LLM 생성 멘트."""
    intro: Annotated[str, Field(description=(
        "질문 맥락에 맞춘 자연스러운 도입 1~2문장. 첫 대화면 가벼운 인사를 곁들인다. "
        "검색이면 무엇을·왜 찾았는지(질문 키워드/사용자 상황과 연결지어) 언급하고 몇 건을 찾았는지 알려준다. "
        "추천이면 사용자 질문이나 프로필의 어떤 점을 보고 정책을 골랐는지 짚어준다. "
        "분야 헤더나 정책 나열은 하지 말 것(아래 본문이 담당)."
    ))]
    body_note: Annotated[str, Field(description=(
        "이 답변에 필요한 핵심 멘트(2~3문장 권장). "
        "검색이면 찾은 정책들의 공통점·특징이나 눈여겨볼 점을 짚어주는 짧은 총평, "
        "추천이면 사용자 상황과 각 정책을 연결지어 '왜 이게 적합한지' 구체적인 추천 이유, "
        "비교면 선택 시 고려할 관점·주의점. 제공된 정책명·맥락 범위 안에서만 쓰고 "
        "수치·자격을 지어내지 말 것. 검색·추천에서는 가능하면 채우고, 정말 덧붙일 내용이 없을 때만 빈 문자열로 둔다."
    ))]
    outro: Annotated[str, Field(description=(
        "후속 행동을 유도하는 마무리 1문장(예: 자격 확인·신청 URL 안내·추가 질문 권유). 불필요하면 빈 문자열."
    ))]


_commentary_model = init_chat_model(
    "gpt-4o-mini", model_provider="openai", temperature=0.4
).with_structured_output(_Commentary)


def _build_body_brief(fragments: list[DomainResult]) -> str:
    """LLM 멘트 생성용으로 본문에 담긴 분야·정책명(+짧은 설명)을 요약한다 (사실 범위 고정).

    정책명만 주면 LLM이 자기 학습 지식으로 빈칸을 메워(예: 사용자가 댄 유명 정책명) 검색되지
    않은 내용을 지어낸다. 실제 검색된 정책의 설명을 함께 넘겨 멘트가 그 범위를 벗어나지 않게 한다.
    """
    lines: list[str] = []
    for frag in fragments:
        items: list[str] = []
        for p in frag.get("policies", []):
            nm = p.get("plcyNm", "")
            if not nm:
                continue
            expl = str(p.get("plcyExplnCn") or p.get("plcySprtCn") or "").replace("\n", " ").strip()
            items.append(f"{nm}({expl[:50]}…)" if expl else nm)
        names_txt = ", ".join(items) if items else "(정책명 없음)"
        lines.append(f"- {frag.get('category', '')}: {names_txt}")
    return "\n".join(lines)


async def _generate_commentary(
    state: ShareState, fragments: list[DomainResult], is_first_turn: bool
) -> _Commentary | None:
    """본문 앞뒤 멘트를 LLM으로 생성한다. 실패하면 None(호출부가 결정적 폴백)."""
    inquiry_types = state.get("inquiry_type") or []
    categories = [f.get("category", "") for f in fragments]
    profile = state.get("user_profile") or {}
    profile_hint = "(로그인 프로필 있음 — 사용자 상황에 맞춰 설명)" if profile else "(비로그인 — 일반 안내)"

    user_content = (
        f"[사용자 질문]\n{state.get('user_inquiry', '')}\n\n"
        f"[분야]\n{', '.join(categories) or '미분류'}\n"
        f"[의도]\n{', '.join(inquiry_types) or '검색'}\n"
        f"[첫 대화 여부]\n{'예' if is_first_turn else '아니오(대화 이어짐)'}\n"
        f"[사용자]\n{profile_hint}\n\n"
        f"[답변 본문에 포함된 정책]\n{_build_body_brief(fragments) or '(없음)'}"
    )
    try:
        return cast(_Commentary, await _commentary_model.ainvoke([
            {"role": "system", "content": _COMMENTARY_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]))
    except Exception:
        logger.exception("맥락 멘트 생성 실패 — 결정적 폴백 사용")
        return None


_COMMENTARY_SYSTEM_PROMPT = """당신은 청년정책 챗봇의 답변 편집자입니다.
정책 본문(표·정책 블록)은 이미 시스템이 결정적으로 작성했습니다. 당신은 그 본문 **앞뒤에 붙일 서술 멘트**만 만듭니다.

원칙:
- 따뜻하고 간결한 한국어. 과장·영업 문구 금지.
- [답변 본문에 포함된 정책]에 적힌 정책명·분야 범위 안에서만 말하세요. 금액·자격·마감 등 구체 수치를 새로 지어내지 마세요.
- 의도가 '검색'이면 intro에서 사용자가 무엇을 물었는지·왜 그 정책들을 찾았는지(질문 키워드나 사용자 상황과 연결)와 몇 건을 찾았는지 언급하세요.
  body_note에는 찾은 정책들 사이의 공통점이나 눈여겨볼 특징을 2~3문장으로 짚어주세요(단순 "찾아봤어요" 한 줄로 끝내지 말 것).
- 의도가 '추천'이면 intro에서 사용자 질문/프로필의 어떤 점을 보고 이 정책들을 골랐는지 밝히고,
  body_note에는 정책별로(또는 묶어서) 왜 사용자 상황에 적합한지 구체적인 추천 이유를 2~3문장으로 담으세요.
- 의도가 '비교'면 body_note에 선택 시 고려할 관점·주의점을 담으세요.
- 의도가 '상세조회'면 intro는 [답변 본문에 포함된 정책]에 **실제로 있는 정책명**을 기준으로 "○○에 대한 자세한 정보를 알려드릴게요." 형태로 쓰세요. body_note에는 그 정책의 짧은 핵심 요약이나 신청 시 주의사항 한 마디를 담으세요(1~2문장).
  ※ 사용자가 말한 정책명이 [답변 본문에 포함된 정책] 목록에 **없으면**, 그 이름을 사실인 양 단정해서 반복하지 마세요. 대신 "요청하신 내용과 관련된 정책을 찾아봤어요." 처럼 안내하고, 실제로 찾은 정책명을 언급하세요.
- 정책 블록/표/분야 헤더를 직접 그리지 마세요(중복 방지). 인사는 intro에서 첫 대화일 때만.
- 검색·추천의 intro·body_note는 빈 문자열로 비우지 마세요(이 답변에서 가장 중요한 멘트입니다).
  그 외(비교·상세조회)는 정말 덧붙일 내용이 없을 때만 빈 문자열로 두세요."""


# ── 비교 표 셀 요약(LLM) ──────────────────────────────────────────────────────
# 비교 표의 '지원 내용/대상' 원문은 매우 길어 말줄임표로 잘렸다. 표에 넣기 전에
# 한 번의 배치 LLM 호출로 핵심만 짧은 문구로 요약한다(정책당 호출 X). 실패하면 원문 폴백.
#
# 구조화 출력(with_structured_output)을 쓴다 — 평문 JSON + json.loads는 모델이 값 안에
# 따옴표를 안 이스케이프하면 깨져서(확률적 파싱 실패) 요약이 산발적으로 폴백됐다.

class _CellSummary(BaseModel):
    """정책 1건의 비교 표 셀 요약."""
    idx: Annotated[int, Field(description="입력 정책의 0-based 순번")]
    support: Annotated[str, Field(description="'지원 내용' 요약 (25자 이내, 한 줄, 없으면 빈 문자열)")] = ""
    target: Annotated[str, Field(description="'지원 대상' 요약 (25자 이내, 한 줄, 없으면 빈 문자열)")] = ""


class _CellSummaries(BaseModel):
    """비교 대상 정책들의 셀 요약 묶음."""
    items: list[_CellSummary]


_summary_model = init_chat_model(
    "gpt-4o-mini", model_provider="openai", temperature=0
).with_structured_output(_CellSummaries)

_CELL_SUMMARY_SYSTEM_PROMPT = """당신은 청년정책 비교 표의 셀 요약기입니다.
각 정책의 '지원 내용'과 '지원 대상'을 표에 들어갈 만큼 **짧고 핵심만** 요약하세요.

규칙:
- 각 항목 25자 이내, 한 줄. 금액·기간 등 핵심 수치는 살리세요.
- 원문에 있는 사실만 쓰고 지어내지 마세요. 정보가 없으면 빈 문자열.
- 행정 군더더기(사업 근거·법령 조항 등)는 빼고 실제 혜택/대상만 남기세요.
- 입력으로 받은 모든 정책을 idx 그대로 매겨 빠짐없이 반환하세요."""


async def _summarize_compare_cells(policies: list[dict]) -> dict[str, dict[str, str]]:
    """비교 표의 긴 셀(지원 내용/대상)을 배치 LLM 호출로 짧게 요약한다.

    반환: {정책키: {"plcySprtCn": 요약, "ptcpPrpTrgtCn": 요약}} — build_comparison_table의
    cell_overrides로 넘긴다. 실패하면 빈 dict(원문 그대로 길이 제한 폴백).
    """
    rows = [p for p in policies if p and p.get("plcyNm")]
    if len(rows) < 2:
        return {}

    items = []
    for idx, p in enumerate(rows):
        items.append({
            "idx": idx,
            "plcyNm": p.get("plcyNm", ""),
            "support": str(p.get("plcySprtCn") or ""),
            "target": str(p.get("ptcpPrpTrgtCn") or ""),
        })

    try:
        result = cast(_CellSummaries, await _summary_model.ainvoke([
            SystemMessage(content=_CELL_SUMMARY_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(items, ensure_ascii=False)),
        ]))
    except Exception:
        logger.exception("비교 표 셀 요약 실패 — 원문 폴백")
        return {}

    overrides: dict[str, dict[str, str]] = {}
    for entry in result.items:
        if not (0 <= entry.idx < len(rows)):
            continue
        cell: dict[str, str] = {}
        if entry.support.strip():
            cell["plcySprtCn"] = entry.support.strip()
        if entry.target.strip():
            cell["ptcpPrpTrgtCn"] = entry.target.strip()
        if cell:
            overrides[policy_key(rows[entry.idx])] = cell
    logger.info("비교 표 셀 요약 완료 — %d/%d건", len(overrides), len(rows))
    return overrides


async def composer_node(state: ShareState) -> dict:
    """도메인 fragment를 모아 일관된 최종 답변으로 조립한다.

    분기:
    - 활성 결과 0개 → fallback (첫 턴이면 인사 포함)
    - "비교" ∈ inquiry_type & 정책 2개 이상 → 결정적 통합 비교 표 (분야 교차 가능)
    - 1개 → 분야 헤더 + 본문
    - N개 → 도입부 + 분야마다 헤더 + 본문
    도입부·맥락 멘트·맺음말은 LLM이 생성하고(실패 시 결정적 폴백), 본문은 결정적으로 조립한다.
    """
    logger.info(
        "composer 노드 실행 — category=%s, inquiry_type=%s",
        state.get("category"), state.get("inquiry_type"),
    )

    suggestions = state.get("suggestions", [])
    fragments = _collect_active_fragments(state)

    # 대화기록(messages)이 비어 있으면 첫 턴 → 인사 템플릿 부착.
    is_first_turn = not state.get("messages")

    if not fragments:
        body = _FALLBACK_MESSAGE
        if is_first_turn:
            body = f"{_GREETING_TEMPLATE}\n\n{body}"
        return {"final_response": body, "suggestions": suggestions}

    inquiry_types = state.get("inquiry_type") or []

    # 상세조회 단독: 프론트가 policies 카드로 렌더하므로 composer는 짧은 안내 멘트만 생성한다.
    if inquiry_types == ["상세조회"]:
        # intro는 **실제로 검색된 대표 정책명**으로 결정적으로 만든다.
        # (LLM에 맡기면 사용자가 말한 정책명을 그대로 받아써, DB에 없는 정책도 "찾은 척"하는
        #  환각이 난다 — 예: "청년도약계좌" 요청에 무관한 재무상담만 검색됐는데도 청년도약계좌라 단정.)
        primary_name = next(
            (p["plcyNm"] for frag in fragments for p in frag.get("policies", []) if p.get("plcyNm")),
            "",
        )
        commentary = await _generate_commentary(state, fragments, is_first_turn)
        parts: list[str] = []
        greeting = f"{_GREETING_TEMPLATE}\n\n" if is_first_turn else ""
        if primary_name:
            parts.append(f"{greeting}**{primary_name}**에 대한 자세한 정보를 알려드릴게요.")
        elif commentary and commentary.intro.strip():
            parts.append(f"{greeting}{commentary.intro.strip()}")
        elif is_first_turn:
            parts.append(_GREETING_TEMPLATE.rstrip())
        # body_note(요약·주의사항)는 검색된 정책 범위에서만 — _build_body_brief가 설명을 함께 넘겨 근거를 고정.
        if commentary and commentary.body_note.strip():
            parts.append("")
            parts.append(commentary.body_note.strip())
        final_response = "\n".join(parts).rstrip() + "\n" if parts else ""
        return {"final_response": final_response, "suggestions": suggestions}

    # 비교 의도 → 분야 fragment를 쌓는 대신 raw 정책 메타로 통합 비교 표를 결정적으로 조립.
    # (주거+일자리처럼 분야가 섞여도 하나의 표로 나란히 비교. 정책 2개 미만이면 일반 분기로 폴백.)
    comparison_table = ""
    if "비교" in inquiry_types:
        # 후속질문이 직전 정책을 지정했으면(resolved_policies) 검색 결과 대신 '그 정책들'로 표를 만든다.
        # → "공공근로와 미래일자리 비교"가 유사도 top-k가 아니라 정확히 지정 2개만 비교됨.
        resolved = state.get("resolved_policies") or []
        compare_policies = resolved if len(resolved) >= 2 else _collect_policies_for_compare(fragments)
        # 긴 '지원 내용/대상' 셀은 LLM이 한 번에 짧게 요약 → 말줄임표 잘림 대신 핵심 문구로.
        cell_overrides = await _summarize_compare_cells(compare_policies)
        comparison_table = build_comparison_table(compare_policies, cell_overrides=cell_overrides)

    # 앞뒤 맥락 멘트(추천 이유·비교 관점 등)는 LLM이, 본문은 결정적으로 — 실패 시 None 폴백.
    commentary = await _generate_commentary(state, fragments, is_first_turn)

    parts: list[str] = []

    # 도입부: LLM intro 우선. 실패 시 첫 턴 인사 + (멀티 분야면) 결정적 도입.
    if commentary and commentary.intro.strip():
        parts.append(commentary.intro.strip())
        parts.append("")
    elif is_first_turn:
        parts.append(_GREETING_TEMPLATE)
        parts.append("")

    if comparison_table:
        if not (commentary and commentary.intro.strip()):
            categories = [f["category"] for f in fragments]
            parts.append(f"{', '.join(categories)} 분야 정책을 비교해 드릴게요." if len(categories) > 1
                         else f"{categories[0]} 정책을 비교해 드릴게요.")
            parts.append("")
        parts.append(comparison_table)
        # 하이브리드: 결정적 표 아래에 각 도메인 에이전트의 정성 코멘트 배치.
        domain_comments = _build_domain_comments(fragments)
        if domain_comments:
            parts.append("")
            parts.append(domain_comments)
    else:
        if not (commentary and commentary.intro.strip()) and len(fragments) > 1:
            parts.append(_build_intro([f["category"] for f in fragments]))
            parts.append("")

        for frag in fragments:
            parts.append(f"## {frag['category']}")
            parts.append("")
            parts.append(_strip_boilerplate(frag["text"]))
            parts.append("")

    # 본문 뒤 맥락 멘트(추천 이유·비교 관점 등) + 맺음말.
    if commentary and commentary.body_note.strip():
        parts.append(commentary.body_note.strip())
        parts.append("")
    if commentary and commentary.outro.strip():
        parts.append(commentary.outro.strip())
        parts.append("")

    final_response = "\n".join(parts).rstrip() + "\n"
    return {"final_response": final_response, "suggestions": suggestions}
