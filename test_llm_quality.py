# -*- coding: utf-8 -*-
"""LLM 응답 품질 / 멀티턴 후속질문 통합 테스트 하네스.

실제 컨트롤러 흐름을 모사한다:
  - 한 대화 세션 = 고정 thread_id + 누적 messages 히스토리
  - 매 턴: supervisor.run(messages=지금까지 히스토리, thread_id=세션키)
  - 턴 종료 후: (user, bot full_message)를 히스토리에 append → 다음 턴의 맥락이 됨

후속질문("두번째 거", "그거 자세히", "A랑 B 비교")은 thread_id 기반 정책 메모리와
messages 히스토리 둘 다 있어야 동작하므로, 세션 단위로 상태를 유지하며 검증한다.

실행: venv\\Scripts\\python.exe test_llm_quality.py
"""
import asyncio
import selectors
import sys
import io

from dotenv import load_dotenv
load_dotenv()

# stdout UTF-8 강제 (Windows 콘솔 인코딩 방어)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from api.chat_service.langgraph.supervisor import ChatbotSupervisor

# 추천(자격 진단) 테스트용 가짜 로그인 프로필
SAMPLE_PROFILE = {
    "user_id": 4,
    "birth_date": "2000-05-10",
    "zipcd": "전북특별자치도",
    "mrgsttscd": "미혼",
}

SEP = "=" * 78
SUB = "-" * 78


class Session:
    """한 대화 세션. messages 히스토리와 thread_id를 유지하며 멀티턴을 흘려보낸다."""

    def __init__(self, sup: ChatbotSupervisor, thread_id: str,
                 role: str = "guest", profile: dict | None = None):
        self.sup = sup
        self.thread_id = thread_id
        self.role = role
        self.profile = profile
        self.messages: list[dict] = []   # 누적 히스토리 (controller의 get_history 모사)

    async def ask(self, text: str, note: str = "") -> dict:
        print(f"\n{SUB}")
        print(f"🧑 사용자: {text}" + (f"   « {note}" if note else ""))
        result = await self.sup.run(
            user_inquiry=text,
            user_role=self.role,
            user_profile=self.profile,
            messages=list(self.messages),   # 이번 턴 이전까지의 히스토리
            thread_id=self.thread_id,
        )
        cat = result.get("category")
        itype = result.get("inquiry_type")
        msg = result.get("message", "")
        policies = result.get("policies", []) or []
        suggestions = result.get("suggestions", []) or []
        full = result.get("full_message", "")

        print(f"   [분야={cat} | 의도={itype} | policies={len(policies)}건 | suggestions={len(suggestions)}개]")
        print(f"🤖 메시지:\n{msg if msg else '   (빈 message)'}")
        if policies:
            names = [p.get("plcyNm", "?") for p in policies]
            print(f"   📋 정책 카드: {names}")
        if suggestions:
            print(f"   💬 추천 후속질문: {suggestions}")

        # 히스토리 갱신 — controller는 full_message(정형 전 전체 답변)를 저장한다.
        history_bot = full or msg
        self.messages.append({"role": "user", "content": text})
        self.messages.append({"role": "assistant", "content": history_bot})
        return result


async def conv_A_search_then_followups(sup):
    """대화 A — 검색 → 후속 상세조회(순번 참조) → 후속 비교(순번 참조). 게스트."""
    print(f"\n{SEP}\n[대화 A] 검색 → 후속 상세조회 → 후속 비교  (게스트)\n{SEP}")
    s = Session(sup, "testA-guest", role="guest")
    await s.ask("청년 월세 지원 정책 알려줘", note="검색")
    await s.ask("두번째 정책 자세히 알려줘", note="후속/상세조회·순번참조")
    await s.ask("첫번째랑 두번째 비교해줘", note="후속/비교·순번참조")


async def conv_B_general_then_policy(sup):
    """대화 B — 일반대화(인사/감사) → 정책으로 전환 → 후속. 게스트."""
    print(f"\n{SEP}\n[대화 B] 일반대화 → 정책 전환 → 후속  (게스트)\n{SEP}")
    s = Session(sup, "testB-guest", role="guest")
    await s.ask("안녕!", note="일반대화")
    await s.ask("너는 뭘 도와줄 수 있어?", note="일반대화/능력질문")
    await s.ask("청년 취업 지원 정책 뭐 있어?", note="검색/전환")
    await s.ask("그 중에 첫번째 거 좀 더 알려줘", note="후속/상세조회")


async def conv_C_multi_category(sup):
    """대화 C — 멀티 분야 검색 + 후속 비교. 게스트."""
    print(f"\n{SEP}\n[대화 C] 멀티 분야 검색 → 후속 비교  (게스트)\n{SEP}")
    s = Session(sup, "testC-guest", role="guest")
    await s.ask("주거랑 일자리 정책 같이 알려줘", note="검색/멀티분야")
    await s.ask("방금 알려준 것들 비교해줘", note="후속/비교·멀티분야")


async def conv_D_detail_direct(sup):
    """대화 D — 특정 정책 직접 상세조회 (composer 멘트 신규 추가분 검증). 게스트."""
    print(f"\n{SEP}\n[대화 D] 특정 정책 직접 상세조회  (게스트)\n{SEP}")
    s = Session(sup, "testD-guest", role="guest")
    await s.ask("청년도약계좌에 대해 자세히 알려줘", note="상세조회·직접지정")
    await s.ask("신청 자격이 어떻게 돼?", note="후속/같은 정책 심화")


async def conv_E_recommend_login(sup):
    """대화 E — 로그인 유저 추천(자격 진단) + 후속. user."""
    print(f"\n{SEP}\n[대화 E] 로그인 유저 추천(자격 진단) → 후속  (user)\n{SEP}")
    s = Session(sup, "testE-user", role="user", profile=SAMPLE_PROFILE)
    await s.ask("내 조건에 맞는 주거 정책 추천해줘", note="추천/자격진단")
    await s.ask("세번째 정책 신청 방법 알려줘", note="후속/상세조회")


async def conv_F_multi_intent_count(sup):
    """대화 F — 복합 의도 + 개수 지정. 게스트."""
    print(f"\n{SEP}\n[대화 F] 복합 의도/개수 지정  (게스트)\n{SEP}")
    s = Session(sup, "testF-guest", role="guest")
    await s.ask("교육 지원 정책 3개만 추천하고 비교해줘", note="복합의도(추천+비교)/개수=3")


async def conv_G_no_result_and_edge(sup):
    """대화 G — 결과 없을 법한 질문 / 엉뚱한 후속 (엣지)."""
    print(f"\n{SEP}\n[대화 G] 엣지 — 모호/무관 질문  (게스트)\n{SEP}")
    s = Session(sup, "testG-guest", role="guest")
    await s.ask("고마워 잘 쓸게", note="일반대화/감사")
    await s.ask("복지 문화 정책 알려줘", note="검색")
    await s.ask("오늘 날씨 어때?", note="일반대화/주제이탈")


async def main():
    sup = ChatbotSupervisor()
    print("✅ Supervisor 빌드 완료 — 테스트 시작\n")

    runners = [
        conv_A_search_then_followups,
        conv_B_general_then_policy,
        conv_C_multi_category,
        conv_D_detail_direct,
        conv_E_recommend_login,
        conv_F_multi_intent_count,
        conv_G_no_result_and_edge,
    ]
    for run in runners:
        try:
            await run(sup)
        except Exception as e:
            import traceback
            print(f"\n❌ [{run.__name__}] 예외 발생: {e}")
            traceback.print_exc()

    print(f"\n{SEP}\n✅ 전체 테스트 완료\n{SEP}")


if __name__ == "__main__":
    # Windows psycopg async 요구사항: SelectorEventLoop
    loop = asyncio.SelectorEventLoop(selectors.SelectSelector())
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
