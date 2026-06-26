"""직전 턴 정책 메모리 — 후속질문("두번째 정책", "공공근로와 비교") 해소용 구조화 캐시.

대화 checkpointer(history_service)는 messages 텍스트만 저장한다. 후속질문이 직전에 보여준
정책을 순번/이름으로 가리키려면, 직전 턴의 raw 정책 메타(plcyNm·plcyNo 포함)를 그대로
기억해 둬야 한다. 이 모듈이 thread_id별로 그 리스트를 보관한다.

한계(의도된 단순화): 프로세스 인메모리 저장이라 서버 재시작/멀티워커 간에는 휘발된다.
회원(Postgres) 영속화는 후속 과제(별도 테이블). 단일 워커 개발/시연 환경에는 충분.
"""
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)

# thread_id → 직전 턴 정책 메타 리스트 (LRU). 최근 접근 thread를 끝으로 보낸다.
_store: "OrderedDict[str, list[dict]]" = OrderedDict()

# thread별 보관 정책 수 상한 (직전 턴 결과만 필요하므로 작게)
_MAX_POLICIES = 20
# 보관할 thread 수 상한 — 초과 시 가장 오래 안 쓴 thread부터 제거(LRU).
# 인메모리 구현이라 무한 증가를 막기 위한 임시 안전장치(Postgres 영속화 전까지).
_MAX_THREADS = 500


def get_last_policies(thread_id: str) -> list[dict]:
    """thread의 직전 턴 정책 메타를 반환 (없으면 빈 리스트). 접근 시 LRU 갱신."""
    policies = _store.get(thread_id)
    if policies is None:
        return []
    _store.move_to_end(thread_id)  # 최근 사용 표시
    return policies


def save_last_policies(thread_id: str, policies: list[dict]) -> None:
    """이번 턴이 보여준 정책 메타를 thread에 저장. 빈 결과면 직전 값을 지운다."""
    if not policies:
        _store.pop(thread_id, None)
        return
    _store[thread_id] = policies[:_MAX_POLICIES]
    _store.move_to_end(thread_id)
    # 상한 초과 시 가장 오래된 thread부터 제거
    while len(_store) > _MAX_THREADS:
        evicted, _ = _store.popitem(last=False)
        logger.info("정책 메모리 상한 초과 — 오래된 thread 제거: %s", evicted)
    logger.info("[%s] 직전 정책 메모리 저장 — %d건", thread_id, len(_store[thread_id]))


def clear_last_policies(thread_id: str) -> None:
    """thread의 정책 메모리 삭제 (대화 기록 삭제와 함께 호출용)."""
    _store.pop(thread_id, None)
