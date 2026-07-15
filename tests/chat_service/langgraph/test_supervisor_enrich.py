"""_enrich_policies_support 특성화(characterization) 테스트.

이 테스트는 batch 리팩토링 이전의 관찰 가능한 동작(observable behavior)을 고정한다.
_summarize_support_content의 내부 구현이 아니라 _enrich_policies_support의
입출력 계약만 검증한다 — 리팩토링 후에도 이 파일의 어서션은 그대로 통과해야 한다.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")
os.environ.setdefault(
    "DATABASE_URL_SQLALCHEMY", "postgresql+psycopg://user:pass@localhost:5432/test"
)

import asyncio
import json
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Iterator
from unittest.mock import AsyncMock, patch

from api.chat_service.langgraph import supervisor


@contextmanager
def mock_llm_summary(
    *, return_value: str | None = None, side_effect: BaseException | None = None
) -> Iterator[AsyncMock]:
    """_enrich_policies_support가 사용하는 배치 LLM 호출 지점을 모킹한다.

    리팩토링으로 호출 형태(단건 -> batch)가 바뀌어도 이 함수만 수정하면 되고,
    나머지 테스트의 어서션은 그대로 유지된다. return_value는 배치에 포함된
    모든 idx에 동일하게 적용된다(요청 payload를 파싱해 idx를 그대로 매핑).
    """
    mock_ainvoke = AsyncMock()
    if side_effect is not None:
        mock_ainvoke.side_effect = side_effect
    else:
        async def _fake_ainvoke(messages: list) -> supervisor._SupportSummaries:
            payload = json.loads(messages[1].content)
            items = [
                supervisor._SupportSummary(idx=entry["idx"], summary=return_value or "")
                for entry in payload
            ]
            return supervisor._SupportSummaries(items=items)

        mock_ainvoke.side_effect = _fake_ainvoke
    fake_model = SimpleNamespace(ainvoke=mock_ainvoke)
    with patch.object(supervisor, "_support_summary_model", fake_model):
        yield mock_ainvoke


def test_empty_input_returns_empty_list_without_llm_call() -> None:
    with mock_llm_summary(return_value="요약") as mock_ainvoke:
        result = asyncio.run(supervisor._enrich_policies_support([]))

    assert result == []
    mock_ainvoke.assert_not_called()


def test_summary_added_for_policy_with_support_content() -> None:
    policy = {
        "plcyNo": "P001",
        "plcyNm": "청년 월세 지원",
        "plcySprtCn": "월 20만원을 최대 12개월간 지원합니다.",
    }

    with mock_llm_summary(return_value="월 20만원을 지원하는 정책입니다."):
        result = asyncio.run(supervisor._enrich_policies_support([policy]))

    assert len(result) == 1
    enriched = result[0]
    assert enriched["plcySprtCnSummary"] == "월 20만원을 지원하는 정책입니다."
    assert enriched["plcyNo"] == policy["plcyNo"]
    assert enriched["plcyNm"] == policy["plcyNm"]
    assert enriched["plcySprtCn"] == policy["plcySprtCn"]


def test_missing_or_blank_support_content_gets_no_summary_key() -> None:
    policies = [
        {"plcyNm": "정책A", "plcySprtCn": ""},
        {"plcyNm": "정책B", "plcySprtCn": "   "},
        {"plcyNm": "정책C"},
    ]

    with mock_llm_summary(return_value="요약문") as mock_ainvoke:
        result = asyncio.run(supervisor._enrich_policies_support(policies))

    assert len(result) == 3
    for enriched in result:
        assert "plcySprtCnSummary" not in enriched
    mock_ainvoke.assert_not_called()


def test_llm_failure_returns_policies_without_summary_and_never_raises() -> None:
    policies = [
        {"plcyNm": "정책A", "plcySprtCn": "지원 내용 A"},
        {"plcyNm": "정책B", "plcySprtCn": "지원 내용 B"},
    ]

    with mock_llm_summary(side_effect=RuntimeError("LLM 호출 실패")):
        result = asyncio.run(supervisor._enrich_policies_support(policies))

    assert len(result) == 2
    for original, enriched in zip(policies, result):
        assert "plcySprtCnSummary" not in enriched
        assert enriched["plcyNm"] == original["plcyNm"]


def test_input_dicts_not_mutated() -> None:
    policy = {"plcyNm": "정책A", "plcySprtCn": "지원 내용"}
    policies = [policy]

    with mock_llm_summary(return_value="요약문"):
        result = asyncio.run(supervisor._enrich_policies_support(policies))

    assert "plcySprtCnSummary" not in policy
    assert result[0] is not policy
    assert "plcySprtCnSummary" in result[0]


def test_order_preserved() -> None:
    policies = [
        {"plcyNm": f"정책{i}", "plcySprtCn": f"지원 내용 {i}"} for i in range(5)
    ]

    with mock_llm_summary(return_value="요약"):
        result = asyncio.run(supervisor._enrich_policies_support(policies))

    assert [p["plcyNm"] for p in result] == [p["plcyNm"] for p in policies]


def test_multiple_policies_trigger_exactly_one_batch_call() -> None:
    policies = [
        {"plcyNm": f"정책{i}", "plcySprtCn": f"지원 내용 {i}"} for i in range(5)
    ]

    with mock_llm_summary(return_value="요약") as mock_ainvoke:
        result = asyncio.run(supervisor._enrich_policies_support(policies))

    assert mock_ainvoke.call_count == 1
    assert len(result) == 5
    for enriched in result:
        assert enriched["plcySprtCnSummary"] == "요약"
