"""직전 턴 정책 메모리 — 후속질문("두번째 정책", "공공근로와 비교") 해소용 구조화 캐시.

대화 checkpointer(history_service)는 messages 텍스트만 저장한다. 후속질문이 직전에 보여준
정책을 순번/이름으로 가리키려면, 직전 턴의 raw 정책 메타(plcyNm·plcyNo 포함)를 그대로
기억해 둬야 한다. 이 모듈이 thread_id별로 그 리스트를 보관한다.

한계(의도된 단순화): 프로세스 인메모리 저장이라 서버 재시작/멀티워커 간에는 휘발된다.
회원(Postgres) 영속화는 후속 과제(별도 테이블). 단일 워커 개발/시연 환경에는 충분.
"""
import logging

logger = logging.getLogger(__name__)

# thread_id → 직전 턴 정책 메타 리스트
_store: dict[str, list[dict]] = {}

# thread별 보관 정책 수 상한 (직전 턴 결과만 필요하므로 작게)
_MAX_POLICIES = 20


def get_last_policies(thread_id: str) -> list[dict]:
    """thread의 직전 턴 정책 메타를 반환 (없으면 빈 리스트)."""
    return _store.get(thread_id, [])


def save_last_policies(thread_id: str, policies: list[dict]) -> None:
    """이번 턴이 보여준 정책 메타를 thread에 저장. 빈 결과면 직전 값을 지운다."""
    if not policies:
        _store.pop(thread_id, None)
        return
    _store[thread_id] = policies[:_MAX_POLICIES]
    logger.info("[%s] 직전 정책 메모리 저장 — %d건", thread_id, len(_store[thread_id]))


def clear_last_policies(thread_id: str) -> None:
    """thread의 정책 메모리 삭제 (대화 기록 삭제와 함께 호출용)."""
    _store.pop(thread_id, None)
