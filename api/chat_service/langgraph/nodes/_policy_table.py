# api/chat_service/langgraph/nodes/_policy_table.py
"""정책 비교 표 빌더 — raw 정책 메타로 결정적(LLM 미사용) 마크다운 표를 만든다.

설계 의도 (분야 교차 비교):
- 기존 비교는 도메인 에이전트가 각자 LLM으로 표를 그려서 분야별로 따로 나오고, 형식·정확성이
  매 호출 달라졌다. 이 모듈은 supervisor가 모은 raw 정책 메타(POLICY_METADATA_FIELDS)만으로
  "정책=행, 비교 항목=열" 표를 결정적으로 조립한다.
  → 토큰 0, 변동성 0, 주거+일자리처럼 분야가 섞여도 하나의 표로 나란히 비교.

표 방향을 "정책=행"으로 둔 이유: 정책=열로 두면(교육 에이전트 방식) 정책이 5개 이상이거나
멀티 분야면 열이 폭주해 가독성이 깨진다. 행으로 쌓으면 N개여도 안전하다.
"""
import re
from datetime import date

from ..tools.dday import dday_label, parse_dates

# 비교 표 컬럼 정의: (헤더, 정책 dict 키, 셀 최대 길이). None 키는 별도 계산(분야/마감).
_COLUMNS: tuple[tuple[str, str | None, int], ...] = (
    ("정책명", "plcyNm", 40),
    ("분야", "category", 12),
    ("지원 내용", "plcySprtCn", 45),
    ("지원 대상", "ptcpPrpTrgtCn", 35),
    ("신청 기간", "_apply_period", 28),
    ("마감", "_dday", 22),
    ("신청 URL", "aplyUrlAddr", 60),
)

# 표가 과도하게 길어지지 않도록 비교 정책 수 상한. 초과분은 표 아래 안내.
_MAX_ROWS = 8

_EMPTY = "정보 없음"


def _apply_period(policy: dict) -> str:
    """신청 기간 셀 값 — 상시/마감 구분이 우선, 없으면 aplyYmd 원문."""
    se = (policy.get("aplyPrdSeCd") or "").strip()
    if se == "상시":
        return "상시"
    ymd = (policy.get("aplyYmd") or "").strip()
    return ymd or _EMPTY


def _cell(value, limit: int) -> str:
    """마크다운 표 셀로 안전하게 정리한다 (파이프/개행 제거 + 길이 제한)."""
    text = "" if value is None else str(value)
    text = text.replace("|", "/").replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if not text or text in ("None", "-"):
        return _EMPTY
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def _resolve(policy: dict, key: str | None) -> str:
    """컬럼 키 → 셀 원본 값. 계산 컬럼(_dday/_apply_period)은 파생."""
    if key == "_dday":
        return dday_label(
            policy.get("aplyPrdSeCd", ""),
            policy.get("aplyYmd", ""),
            policy.get("bizPrdEndYmd", ""),
        )
    if key == "_apply_period":
        return _apply_period(policy)
    return policy.get(key, "")  # type: ignore[arg-type]


def build_comparison_table(policies: list[dict]) -> str:
    """raw 정책 메타 리스트로 마크다운 비교 표 + 결정적 요약을 만든다.

    정책이 2개 미만이면 빈 문자열을 반환한다(비교 불가 — 호출부가 일반 분기로 폴백).
    """
    rows = [p for p in policies if p and p.get("plcyNm")]
    if len(rows) < 2:
        return ""

    truncated = len(rows) > _MAX_ROWS
    rows = rows[:_MAX_ROWS]

    header = "| " + " | ".join(h for h, _, _ in _COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in _COLUMNS) + " |"
    lines = [header, sep]
    for policy in rows:
        cells = [_cell(_resolve(policy, key), limit) for _, key, limit in _COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")

    table = "\n".join(lines)
    summary = _build_summary(rows)
    parts = [table, "", summary]
    if truncated:
        parts.append("")
        parts.append(f"> 관련 정책이 많아 상위 {_MAX_ROWS}개만 표로 비교했습니다.")
    return "\n".join(parts).rstrip() + "\n"


def _build_summary(policies: list[dict]) -> str:
    """표 아래 결정적 핵심 요약 — 분야 분포 + 가장 임박한 마감."""
    categories = []
    for p in policies:
        c = (p.get("category") or "").strip()
        if c and c not in categories:
            categories.append(c)

    # 마감일이 가장 가까운(미래) 정책 찾기
    soonest: tuple[int, str] | None = None
    for p in policies:
        dates = parse_dates(f"{p.get('aplyYmd', '') or ''} {p.get('bizPrdEndYmd', '') or ''}")
        future = [d for d in dates if (d - date.today()).days >= 0]
        if not future:
            continue
        delta = (min(future) - date.today()).days
        if soonest is None or delta < soonest[0]:
            soonest = (delta, p.get("plcyNm", ""))

    bullets = ["**핵심 비교 요약**"]
    if len(categories) > 1:
        bullets.append(f"- 분야: {', '.join(categories)} ({len(categories)}개 분야 교차 비교)")
    if soonest is not None:
        d_txt = "오늘 마감" if soonest[0] == 0 else f"D-{soonest[0]}"
        bullets.append(f"- 마감 임박: **{soonest[1]}** ({d_txt}) — 가장 먼저 신청하세요.")
    bullets.append("- 자세한 자격·금액은 각 정책의 신청 URL에서 확인하세요.")
    return "\n".join(bullets)
