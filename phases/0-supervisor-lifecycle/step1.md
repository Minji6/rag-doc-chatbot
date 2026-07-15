# Step 1: supervisor-singleton

## Files to read

Read these files first to understand the project's architecture and design intent:

- `/docs/ARCHITECTURE.md`
- `/docs/ADR.md`
- `/api/chat_service/langgraph/supervisor.py` (the file to change — read the whole file)
- `/api/chat_service/controller.py` (the consumer of `ChatbotSupervisorDep`)

## Task

**This step is a refactor (behavior-preserving): eliminate per-request StateGraph recompilation.**

Current problem: at the bottom of `api/chat_service/langgraph/supervisor.py`:

```python
ChatbotSupervisorDep = Annotated[ChatbotSupervisor, Depends(ChatbotSupervisor)]
```

`Depends(ChatbotSupervisor)` constructs a **new** `ChatbotSupervisor` on **every request**. Its `__init__` calls `self._build()`, which rebuilds and re-`compile()`s the entire `StateGraph` each time — pure wasted work on the hot path of every chat request.

Why sharing one instance is safe (state the same reasoning in your commit if useful):
- The compiled graph is stateless between invocations; all per-request state flows through `ShareState` passed to `workflow.ainvoke(...)` inside `run()`.
- `ChatbotSupervisor` instance attributes (`self.logger`, `self.workflow`) are set once in `__init__` and never mutated afterwards.
- No checkpointer is attached at compile time (`graph.compile()` with no args), so there is no shared mutable checkpoint state on the instance.

Do the following:

1. In `supervisor.py`, create the singleton and change the dependency so the instance is constructed once per process. Use the standard FastAPI idiom, e.g.:

```python
from functools import lru_cache

@lru_cache(maxsize=1)
def get_chatbot_supervisor() -> ChatbotSupervisor:
    return ChatbotSupervisor()

ChatbotSupervisorDep = Annotated[ChatbotSupervisor, Depends(get_chatbot_supervisor)]
```

   (A module-level `_supervisor = ChatbotSupervisor()` with `Depends(lambda: _supervisor)` is also acceptable, but the `lru_cache` getter is preferred: it keeps construction lazy so merely importing the module doesn't build the graph.)

2. Do not change `ChatbotSupervisor`'s public interface: `run(...)`'s signature and return shape must stay identical. `api/chat_service/controller.py` must keep working with the type annotation `ChatbotSupervisorDep` unchanged at its use site.

## Acceptance Criteria

```bash
python -c "import main"                       # app loads
python -m pytest scripts/test_execute.py -q   # harness safety-net tests still pass
python -m mypy api/chat_service/langgraph/supervisor.py --ignore-missing-imports  # no new type errors in the changed file
```

Plus a singleton proof, WITHOUT needing a DB or API keys — for example:

```bash
python -c "
from api.chat_service.langgraph.supervisor import get_chatbot_supervisor
a = get_chatbot_supervisor(); b = get_chatbot_supervisor()
assert a is b, 'supervisor must be a singleton'
print('singleton OK')
"
```

If importing `supervisor.py` fails due to missing environment (e.g. `OPENAI_API_KEY` required at import time by `init_chat_model`), do NOT try to work around it by restructuring unrelated code: set the step to `"blocked"` with the exact error in `blocked_reason`.

## Verification steps

1. Run the AC commands above.
2. Check the architecture checklist:
   - Graph topology unchanged? (`_build()` internals must be untouched)
   - CLAUDE.md CRITICAL rules respected?
   - API response shape unchanged? (No response-affecting code was modified)
3. Update `phases/0-supervisor-lifecycle/index.json` step 1:
   - Success → `"status": "completed"`, `"summary"`: name the new getter and confirm single construction.
   - Failure after 3 attempts → `"status": "error"`, `"error_message"` with specifics.
   - Environment-blocked (see above) → `"status": "blocked"` + `blocked_reason`.

## Do not

- Do NOT modify `_build()`'s node/edge wiring in any way. Reason: CLAUDE.md CRITICAL — graph topology changes require a new ADR.
- Do NOT change `run(...)`'s signature, defaults, or return dict shape. Reason: the controller and frontend depend on it; this step is behavior-preserving.
- Do NOT introduce a checkpointer, global state, or config changes. Reason: out of scope; lifecycle fix only.
- Do NOT edit `api/chat_service/controller.py` unless strictly required by the dependency change (the `ChatbotSupervisorDep` annotation should keep working as-is).
- Do NOT break existing tests.
