# Step 0: enrich-characterization-test

## Files to read

Read these files first to understand the project's architecture and design intent:

- `/docs/ARCHITECTURE.md`
- `/CLAUDE.md`
- `/api/chat_service/langgraph/supervisor.py` — focus on `_summarize_support_content` and `_enrich_policies_support` (top of file)

## Task

**This step is test-only (no production code changes): write characterization tests that pin down the current behavior of `_enrich_policies_support` before it gets refactored in the next step.**

Background: `_enrich_policies_support(policies)` currently calls the LLM **once per policy** (via `_summarize_support_content`) to add a `plcySprtCnSummary` field. The next step will refactor it into a single batch LLM call. These tests define the contract that must survive that refactor, so **test the observable behavior of `_enrich_policies_support`, not the internals of `_summarize_support_content`** (which may disappear in the refactor).

Create `tests/chat_service/langgraph/test_supervisor_enrich.py` (create the `tests/` package directories with empty `__init__.py` files as needed) with tests covering this contract:

1. **Empty input**: `_enrich_policies_support([])` returns `[]` without any LLM call.
2. **Summary added**: for a policy with non-empty `plcySprtCn`, the result dict gains `plcySprtCnSummary` with the (mocked) summary text, and the original fields are unchanged.
3. **Empty support content**: a policy whose `plcySprtCn` is missing/empty/whitespace gets **no** `plcySprtCnSummary` key.
4. **LLM failure fallback**: if the LLM call raises, the function still returns the policies (same length, same order) just without `plcySprtCnSummary` — it must never raise.
5. **Input not mutated**: the input list's dicts must not be mutated (the function returns copies).
6. **Order preserved**: output policies are in the same order as input.

Implementation constraints for the tests:

- **Mock every LLM call.** No network/API calls. Patch at the boundary the current code uses (`api.chat_service.langgraph.supervisor._enrich_llm.ainvoke` today), but structure each test so the patch target is easy to change — e.g. a single helper/fixture that installs the mock — because the refactor in step 1 will change the internal call shape. After the refactor, ONLY that helper should need updating, not the assertions.
- `pytest-asyncio` is NOT installed. Run async functions with `asyncio.run(...)` inside normal sync test functions.
- Importing `supervisor.py` requires env vars (OpenAI key etc.). At the top of the test module, set harmless dummy values with `os.environ.setdefault("OPENAI_API_KEY", "test-key")` (and any others the import chain needs) BEFORE importing the module under test. Do not rely on a real `.env`.
- Follow the project's code conventions (snake_case, type hints).

## Acceptance Criteria

```bash
python -m pytest tests/ -q                     # new tests pass
python -m pytest scripts/test_execute.py -q    # harness safety-net still passes
```

## Verification steps

1. Run the AC commands above.
2. Confirm no production code under `api/` was modified (`git status` should show only new files under `tests/`).
3. Update `phases/1-enrich-batch/index.json` step 0:
   - Success → `"status": "completed"`, `"summary"`: name the test file and the behaviors pinned.
   - Failure after 3 attempts → `"status": "error"` + `error_message`.

## Do not

- Do NOT modify any file under `api/`. Reason: this step only pins existing behavior; the refactor happens in step 1.
- Do NOT install any new packages (no pytest-asyncio, no pip install). Reason: requirements.txt must stay unchanged; use `asyncio.run`.
- Do NOT make real LLM/API calls in tests. Reason: tests must run offline and deterministic.
- Do NOT test `_summarize_support_content` directly. Reason: it is an internal detail that the next step may remove.
