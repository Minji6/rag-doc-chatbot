# Step 0: remove-dead-chat-controller

## Files to read

Read these files first to understand the project's architecture and design intent:

- `/docs/ARCHITECTURE.md`
- `/docs/ADR.md`
- `/api/chat/controller.py` (the file to be removed)
- `/main.py` (router registration — note which controllers are actually registered)
- `/api/chat_service/controller.py` (the REAL, live chat router — do NOT touch)

## Task

**This step is a refactor (behavior-preserving): remove dead code.**

`api/chat/controller.py` is dead code. Evidence:

- `main.py` registers only `auth_service`, `history_service`, `chat_service`, and `upload_service` controllers via `app.include_router(...)`. `api.chat.controller` is never imported there or anywhere else.
- A repo-wide grep for `api.chat` imports (excluding `api.chat_service`) finds no references.
- It contains an obsolete early version of the chat endpoint (`/api/chat` with a module-level `ChatbotSupervisor()` singleton), superseded by `api/chat_service/controller.py` (`/api/chat/service`).

Do the following:

1. Re-verify it is unreferenced: search the whole repo (excluding `venv/`, `__pycache__/`, `.claude/`) for imports of `api.chat.controller` or `from api.chat import`. If you find a real reference, STOP and set the step status to `"error"` explaining where — do not delete.
2. Delete the entire `api/chat/` directory (it contains only `controller.py` and possibly `__pycache__`).
3. Verify the app still loads (see AC).

## Acceptance Criteria

```bash
python -c "import main"                       # app module loads without ImportError
python -m pytest scripts/test_execute.py -q   # harness safety-net tests still pass
```

Also verify by search: no remaining references to `api.chat.controller` or `api/chat/` (path with slash, excluding `api/chat_service/`) anywhere outside `venv/` and `.claude/`.

## Verification steps

1. Run the AC commands above.
2. Check the architecture checklist:
   - Does it follow ARCHITECTURE.md's directory structure? (Yes — ARCHITECTURE.md's directory tree does not include `api/chat/`... actually it lists `chat/` as "챗봇 엔드포인트"; if the deletion makes that doc line stale, update `/docs/ARCHITECTURE.md` to remove the `api/chat/` line from the directory tree in the same commit.)
   - Does it respect CLAUDE.md's CRITICAL rules?
3. Update `phases/0-supervisor-lifecycle/index.json` step 0:
   - Success → `"status": "completed"`, `"summary"`: mention that `api/chat/` was deleted and ARCHITECTURE.md updated (if it was).
   - Failure after 3 attempts → `"status": "error"`, `"error_message"` with specifics.

## Do not

- Do NOT touch `api/chat_service/` — similar name, but it is the live service router. Reason: deleting or editing it breaks the actual `/api/chat/service` endpoint.
- Do NOT remove `HistoryInMemoryAgent` (`api/history_service/agent_history_inmemory.py`) even though the dead controller imports it. Reason: it is used elsewhere by the history service for guest sessions.
- Do NOT add any new features or files (other than the doc line fix described above).
- Do NOT break existing tests.
