import logging
from datetime import date

logger = logging.getLogger(__name__)

# 복지 정책 자격 판별에 필요한 필드 정의
# (db_field, 한글 레이블, 질문 문장)
_WELFARE_FIELDS = [
    ("jobcd",       "취업 상태",  "현재 취업 상태가 어떻게 되시나요? (예: 미취업, 재직, 자영업)"),
    ("earncndsecd", "소득 분위",  "소득 분위가 어떻게 되시나요? (1~10분위, 예: 3분위)"),
    ("zipcd",       "거주 지역",  "현재 거주하시는 지역(시/도)이 어디인가요?"),
    ("mrgsttscd",   "혼인 여부",  "혼인 여부가 어떻게 되시나요? (미혼/기혼/이혼)"),
    ("schoolcd",    "학력",       "최종 학력이 어떻게 되시나요? (예: 고졸, 대학 재학, 대학 졸업)"),
    ("sbizcd",      "특수 분류",  "해당되는 특수 분류가 있으신가요? (예: 장애, 한부모, 다문화, 해당 없음)"),
]


def calc_age(birth_date_str: str) -> int | None:
    """'YYYY-MM-DD' 형식 생년월일 → 만 나이 계산."""
    try:
        bd = date.fromisoformat(birth_date_str)
        today = date.today()
        return today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day))
    except (ValueError, TypeError):
        return None


def _is_empty(value) -> bool:
    """None, 빈 문자열, '제한없음'이면 미입력으로 판단."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip() in ("", "제한없음"):
        return True
    return False


def _is_empty_income(value) -> bool:
    """소득 분위는 0도 유효한 값(무소득)이므로 None과 빈 문자열만 미입력으로 판단."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def check_missing_profile(user_profile: dict) -> tuple[list[str], str]:
    """
    user_profile(DB에서 온 dict)에서 복지 정책 판별에 필요한 빈 필드를 확인합니다.

    Returns:
        (missing_fields, follow_up_question)
        - missing_fields: 비어 있는 필드명 목록
        - follow_up_question: 사용자에게 보낼 유도 질문 문자열 (없으면 빈 문자열)
    """
    missing = [
        (field, question)
        for field, _label, question in _WELFARE_FIELDS
        if (
            _is_empty_income(user_profile.get(field))
            if field == "earncndsecd"
            else _is_empty(user_profile.get(field))
        )
    ]

    if not missing:
        return [], ""

    lines = ["더 정확한 복지 정책을 안내해 드리기 위해 몇 가지 여쭤볼게요 😊"]
    for _, question in missing:
        lines.append(f"  • {question}")

    return [f for f, _ in missing], "\n".join(lines)


def build_profile_context(user_role: str, user_profile: dict) -> str:
    """
    system prompt에 삽입할 사용자 프로필 컨텍스트 문자열을 생성합니다.
    guest면 빈 문자열 반환.
    """
    if user_role != "user" or not user_profile:
        return ""

    age = calc_age(user_profile.get("birth_date", ""))

    def v(val):
        return val if not _is_empty(val) else "미입력"

    lines = ["[사용자 프로필]"]
    lines.append(f"  나이        : {age}세" if age else "  나이        : 미입력")
    lines.append(f"  거주 지역   : {v(user_profile.get('zipcd'))}")
    lines.append(f"  취업 상태   : {v(user_profile.get('jobcd'))}")
    lines.append(f"  소득 분위   : {v(user_profile.get('earncndsecd'))}")
    lines.append(f"  혼인 여부   : {v(user_profile.get('mrgsttscd'))}")
    lines.append(f"  학력        : {v(user_profile.get('schoolcd'))}")
    lines.append(f"  전공 분야   : {v(user_profile.get('plcymajorcd'))}")
    lines.append(f"  특수 분류   : {v(user_profile.get('sbizcd'))}")
    lines.append(f"  관심 분야   : {v(user_profile.get('category'))}")

    return "\n".join(lines)
