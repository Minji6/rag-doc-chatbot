"""정책 벡터 검색 공통 인프라 — 4개 도메인 검색 노드가 공유한다.

이전엔 housing/employment/education/welfare 검색 노드가 각자 동일한 PGVector 싱글톤과
pick_policy_fields 함수를 4벌씩 따로 들고 있었다(같은 컬렉션·임베딩·필드). 여기로 모아 1벌로 만든다.
(검색 파라미터 k/threshold·도메인 필터는 각 노드가 호출 시점에 지정하므로 동작은 동일.)
"""
from langchain_postgres import PGVector
from langchain.embeddings import init_embeddings

from api.common.sqlalchemy_conf import engine
from ..constants import PGVECTOR_COLLECTION_NAME, POLICY_METADATA_FIELDS

# 정책 벡터 검색용 PGVector 싱글톤 (import 시점 1회 생성).
# 4개 검색 노드가 동일 인스턴스를 공유 — 임베딩 객체 중복 생성 제거.
vectorstore = PGVector(
    embeddings=init_embeddings("openai:text-embedding-3-large"),
    collection_name=PGVECTOR_COLLECTION_NAME,
    connection=engine,
    async_mode=True,
)


def _is_zero(v) -> bool:
    try:
        return int(v) == 0
    except (TypeError, ValueError):
        return False


def pick_policy_fields(metadata: dict) -> dict:
    """PGVector 메타에서 화이트리스트 필드(POLICY_METADATA_FIELDS)만 추려 dict 생성.

    4개 도메인이 sprtTrgtMinAge/MaxAge/AgeLmtYn을 공유 스키마로 쓰므로 정제도 여기서
    한 번만 한다. sprtTrgtAgeLmtYn != "Y"(연령 제한 없음/불명)이면 min/max age는 의미 없는
    플레이스홀더인 경우가 많다 (check_policy_eligibility의 _check_age도 동일하게 처리).
    실측 결과 lmtYn="Y"인데도 min=max=0으로 채워진(사실상 미기재) 소스 데이터가 있어
    (예: 대학생 대상 장학금에 "0~0세" 제한은 있을 수 없음), 그 경우도 함께 비운다.
    그대로 카드에 노출하면 "만 0~0세"처럼 사용자에게 오해를 준다.
    """
    p = {key: metadata.get(key) for key in POLICY_METADATA_FIELDS}
    min_age, max_age = p.get("sprtTrgtMinAge"), p.get("sprtTrgtMaxAge")
    if p.get("sprtTrgtAgeLmtYn") != "Y" or (_is_zero(min_age) and _is_zero(max_age)):
        p["sprtTrgtMinAge"] = None
        p["sprtTrgtMaxAge"] = None
    return p


def build_profile_query(inquiry: str, user_profile: dict) -> str:
    """추천 의도일 때 유저 프로필을 검색어에 덧붙여 벡터 검색 품질을 높인다 (4개 도메인 공용).

    실제 user_profile 키(auth_service/model.py 기준):
    birth_date, zipcd, schoolcd, plcymajorcd, jobcd, earncndsecd, mrgsttscd, sbizcd

    '제한없음'은 유효한 값으로 그대로 포함한다 — 제한 없는 정책과 매칭되도록. None/빈값만 제외.
    검색어 텍스트만 바꿀 뿐 k(반환 개수)는 건드리지 않는다 — 여전히 최대 k개가 나온다.
    """
    profile_parts = []
    if user_profile.get("birth_date"):
        profile_parts.append(f"생년월일:{user_profile['birth_date']}")
    if user_profile.get("zipcd"):
        profile_parts.append(f"거주지역:{user_profile['zipcd']}")
    if user_profile.get("schoolcd"):
        profile_parts.append(f"학교:{user_profile['schoolcd']}")
    if user_profile.get("plcymajorcd"):
        profile_parts.append(f"전공:{user_profile['plcymajorcd']}")
    if user_profile.get("jobcd"):
        profile_parts.append(f"직업:{user_profile['jobcd']}")
    if user_profile.get("earncndsecd"):
        profile_parts.append(f"소득분위:{user_profile['earncndsecd']}분위")
    if user_profile.get("mrgsttscd"):
        profile_parts.append(f"혼인상태:{user_profile['mrgsttscd']}")
    if user_profile.get("sbizcd"):
        profile_parts.append(f"소상공인업종:{user_profile['sbizcd']}")

    return f"{inquiry} / {' '.join(profile_parts)}" if profile_parts else inquiry


def similarity_to_score(distance: float) -> int:
    """벡터 검색 거리(코사인 distance)를 0~100 적합도(관련도) 점수로 환산한다.

    적합도 = "질문·프로필과 얼마나 의미적으로 가까운 정책이냐"이며, 자격(신청 가능 여부)과는
    별개 축이다. PGVector가 돌려주는 distance는 코사인 거리(작을수록 유사)이고,
    코사인 유사도 = 1 - distance 이므로 이를 백분율로 환산한다.
      distance 0.0 → 100점(완전 일치), 0.3 → 70점, 0.85 → 15점
    검색 임계값(≤0.9) 안에서만 호출되므로 실제 점수는 대략 10~100 범위. [0,100]로 클램프.
    """
    score = round((1.0 - float(distance)) * 100)
    return max(0, min(100, score))
