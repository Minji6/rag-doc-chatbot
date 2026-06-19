import logging
from datetime import date
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool(return_direct=True)
async def compare_policies(policy_list: list[dict]) -> str:
    """
    여러 정책을 한눈에 비교할 수 있는 표 형태로 변환합니다.
    검색 결과가 2개 이상일 때 호출하세요.
    결과는 LLM 재가공 없이 바로 사용자에게 전달됩니다.

    Args:
        policy_list: 비교할 정책 딕셔너리 목록

    Returns:
        str: 정책명, 지원 대상, 혜택, 마감일, 신청 URL을 포함한 비교표 텍스트
    """
    logger.info(f"compare_policies 실행: 정책 수={len(policy_list)}")

    lines = ["=" * 60, "📋 정책 비교표", "=" * 60]

    for idx, policy in enumerate(policy_list, 1):
        lines.append(f"\n[정책 {idx}]")
        lines.append(f"  정책명    : {policy.get('plcyNm', '정보 없음')}")
        lines.append(f"  지원 대상 : {policy.get('addAplyQlfcCndCn', '정보 없음')}")
        lines.append(f"  지원 내용 : {policy.get('plcySprtCn', '정보 없음')}")

        # D-day 표시
        deadline = policy.get('bizPrdEndYmd', '')
        period_type = policy.get('aplyPrdSeCd', '')
        if period_type == "상시":
            dday_str = "상시접수"
        elif deadline:
            today = date.today()
            try:
                deadline_date = date(int(deadline[:4]), int(deadline[4:6]), int(deadline[6:8]))
                diff = (deadline_date - today).days
                if diff < 0:
                    dday_str = "마감"
                elif diff == 0:
                    dday_str = "오늘 마감"
                else:
                    dday_str = f"D-{diff}"
            except (ValueError, IndexError):
                dday_str = deadline
        else:
            dday_str = "정보 없음"

        lines.append(f"  마감일    : {dday_str}")
        apply_url = policy.get('aplyUrlAddr', '')
        lines.append(f"  신청 URL  : {apply_url if apply_url else '정보 없음'}")
        lines.append("-" * 60)

    return "\n".join(lines)


@tool
async def policy_priority_score(user_profile: dict, policy_list: list[dict]) -> str:
    """
    사용자 정보와 정책 조건을 비교해 각 정책의 적합도 점수를 계산하고 순위를 매깁니다.
    최종 답변을 생성하기 직전에 호출해 가장 적합한 정책을 상위에 노출하세요.

    Args:
        user_profile: 사용자 정보 딕셔너리
        policy_list: 점수를 매길 정책 목록

    Returns:
        str: 정책별 점수와 추천 순위 텍스트
    """
    logger.info(f"policy_priority_score 실행: 정책 수={len(policy_list)}")

    user_age = user_profile.get("age")
    today = date.today()
    scored = []

    for policy in policy_list:
        score = 0
        reasons = []

        # 1. 나이 조건 매칭 (30점)
        # sprtTrgtMinAge/sprtTrgtMaxAge가 "0"이면 연령 제한없음을 의미
        min_age_str = policy.get("sprtTrgtMinAge", "0")
        max_age_str = policy.get("sprtTrgtMaxAge", "0")
        min_age = int(min_age_str) if min_age_str else 0
        max_age = int(max_age_str) if max_age_str else 0
        age_ok = True
        if user_age is not None:
            if min_age and min_age > int(user_age):
                age_ok = False
            if max_age and max_age < int(user_age):
                age_ok = False
        if age_ok:
            score += 30
            reasons.append("나이 조건 충족(+30)")
        else:
            reasons.append("나이 조건 불일치(+0)")

        # 2. 마감 임박도 (40점)
        period_type = policy.get("aplyPrdSeCd", "")
        deadline = policy.get("bizPrdEndYmd", "")
        if period_type == "상시":
            score += 20
            reasons.append("상시접수(+20)")
        elif deadline:
            try:
                deadline_date = date(int(deadline[:4]), int(deadline[4:6]), int(deadline[6:8]))
                diff = (deadline_date - today).days
                if diff < 0:
                    reasons.append("마감(+0)")
                elif diff <= 7:
                    score += 40
                    reasons.append(f"마감 임박 D-{diff}(+40)")
                elif diff <= 30:
                    score += 25
                    reasons.append(f"마감 D-{diff}(+25)")
                else:
                    score += 10
                    reasons.append(f"마감 D-{diff}(+10)")
            except (ValueError, IndexError):
                score += 10
                reasons.append("마감일 파싱 불가(+10)")
        else:
            score += 10
            reasons.append("마감일 정보 없음(+10)")

        # 3. 취업 상태 조건 매칭 (30점)
        # jobCd는 한글 텍스트로 저장됨 (예: "제한없음", "미취업자")
        user_job = user_profile.get("job_status", "")
        policy_job = policy.get("jobCd", "제한없음")
        if policy_job == "제한없음" or not policy_job:
            score += 30
            reasons.append("취업 조건 제한없음(+30)")
        elif user_job and user_job == policy_job:
            score += 30
            reasons.append("취업 조건 매칭(+30)")
        else:
            reasons.append("취업 조건 불일치(+0)")

        scored.append({
            "title": policy.get("plcyNm", "정책명 없음"),
            "score": score,
            "reasons": reasons,
        })

    # 점수 내림차순 정렬
    scored.sort(key=lambda x: x["score"], reverse=True)

    lines = ["=" * 60, "🏆 정책 추천 순위 (적합도 점수 기준)", "=" * 60]
    for rank, item in enumerate(scored, 1):
        lines.append(f"\n{rank}위. {item['title']} — {item['score']}점")
        lines.append(f"   근거: {', '.join(item['reasons'])}")
    lines.append("=" * 60)

    return "\n".join(lines)