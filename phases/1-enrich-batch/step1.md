# Step 1: enrich-batch-call

## Files to read

Read these files first to understand the project's architecture and design intent:

- `/docs/ARCHITECTURE.md`
- `/CLAUDE.md`
- `/api/chat_service/langgraph/supervisor.py` — `_summarize_support_content`, `_enrich_policies_support`, and where `_enrich_policies_support` is called in `run()`
- `/api/chat_service/langgraph/nodes/composer_node.py` — **`_summarize_compare_cells` and its `_CellSummary`/`_CellSummaries` models (lines ~236-308). This is the reference implementation: an existing, proven batch-summary pattern in this codebase. Mirror its approach.**
- `/tests/chat_service/langgraph/test_supervisor_enrich.py` — the characterization tests from step 0 that define the contract you must preserve

## Task

**This step is a refactor (behavior-preserving): replace the per-policy LLM calls in `_enrich_policies_support` with ONE batch LLM call.**

Current cost problem: a 상세조회 (detail-inquiry) response with N policies makes N separate LLM calls (`asyncio.gather` over `_summarize_support_content`). N=5 policies → 5 calls. The composer already solved the same problem for comparison-table cells with a single structured-output batch call — reuse that pattern.

Implementation outline (mirror `_summarize_compare_cells`):

1. Define pydantic models in `supervisor.py`, e.g.:

```python
class _SupportSummary(BaseModel):
    idx: int          # 0-based index of the input policy
    summary: str      # 2~3문장 요약, 없으면 빈 문자열

class _SupportSummaries(BaseModel):
    items: list[_SupportSummary]
```

2. Create a module-level structured-output model: `init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0.0).with_structured_output(_SupportSummaries)`.
3. Rewrite `_enrich_policies_support` to:
   - Return `policies` as-is when empty.
   - Build a batch payload of `{idx, plcyNm, plcySprtCn}` for policies with non-empty `plcySprtCn` only (skip empty/whitespace ones — they must not get a summary field).
   - If the batch payload is empty, return copies without summaries.
   - Make ONE `ainvoke` with a system prompt adapted from the existing `_ENRICH_SYSTEM_PROMPT` rules (2~3 sentences, 존댓말, no markdown, no invented facts, plus "입력으로 받은 모든 정책을 idx 그대로 매겨 빠짐없이 반환" like the composer's prompt).
   - On any exception: log with `logger.exception` and return copies WITHOUT `plcySprtCnSummary` (never raise) — same graceful degradation as today.
   - Map results back by `idx` (validate bounds like the composer does), set `plcySprtCnSummary` only for non-empty stripped summaries, return **copies** (`dict(p)`), preserving input order and never mutating inputs.
4. Remove `_summarize_support_content` and, if no longer referenced, the old `_enrich_llm`/`_ENRICH_SYSTEM_PROMPT` bindings (or repurpose the prompt text into the new system prompt).
5. Update the mock-installation helper in `tests/chat_service/langgraph/test_supervisor_enrich.py` to patch the new batch model's `ainvoke`. **Do not weaken or delete any assertion** — the contract tests must pass against the new implementation. Add one new test: N policies with non-empty `plcySprtCn` trigger exactly ONE LLM call.

## Acceptance Criteria

```bash
python -m pytest tests/ -q                     # contract tests + new batch-count test pass
python -m pytest scripts/test_execute.py -q    # harness safety-net still passes
python -m mypy api/chat_service/langgraph/supervisor.py --ignore-missing-imports
```

## Verification steps

1. Run the AC commands above.
2. Check the architecture checklist:
   - Graph topology untouched? (`_build()` must be unmodified)
   - Response shape unchanged? (`plcySprtCnSummary` field semantics identical: present iff a non-empty summary exists)
   - CLAUDE.md CRITICAL rules respected?
3. Update `phases/1-enrich-batch/index.json` step 1:
   - Success → `"status": "completed"`, `"summary"`: state that N-per-policy calls became 1 batch call and tests pass.
   - Failure after 3 attempts → `"status": "error"` + `error_message`.

## Do not

- Do NOT change `_build()` or any graph wiring. Reason: CLAUDE.md CRITICAL — topology changes need a new ADR.
- Do NOT change when `_enrich_policies_support` is invoked in `run()` (still only for 상세조회-단독 with non-empty policies). Reason: behavior-preserving refactor.
- Do NOT touch `composer_node.py`. Reason: it is the reference, not the target.
- Do NOT install new packages. Reason: requirements.txt must stay unchanged.
- Do NOT delete or weaken the step-0 contract assertions. Reason: they are the proof of behavior preservation.
