This project uses the Harness framework. Follow the workflow below.

---

## Workflow

### A. Explore

Read `/docs/` (PRD, ARCHITECTURE, ADR) and `/CLAUDE.md` to understand the project's intent, architecture, and design. Use the Explore agent in parallel if needed.

### B. Discuss

If anything needs clarification or a technical decision before implementation, raise it with the user and discuss. This project's harness is **refactoring-only** (see `docs/PRD.md`) — if you spot a point that would require a behavior change, flag it to the user first and decide whether it needs its own phase.

### C. Design steps

When the user asks for an implementation plan, draft it as a set of steps and ask for feedback.

Design principles:

1. **Minimize scope** — one step touches one layer or module. If multiple modules need changes, split into multiple steps.
2. **Self-containment** — each step file runs in an independent Claude session. Never reference "as discussed earlier" — put everything the session needs directly in the file.
3. **Force pre-reading** — name the relevant doc paths and any files created/modified by prior steps, so the session reads and understands context before acting.
4. **Signature-level instructions** — give interfaces/signatures, leave implementation to the agent. But spell out any hard constraints the design must not violate (graph topology, composer's sole responsibility, whitelist field-name matching, or other CRITICAL rules from `CLAUDE.md`).
5. **AC must be runnable commands** — not "should work," but a real command like `python -m pytest && python -m mypy .`.
6. **Be concrete about pitfalls** — not "be careful," but "don't do X. Reason: Y."
7. **Naming** — step names are kebab-case slugs naming the core module/task (e.g. `supervisor-cleanup`, `tools-dedupe`).
8. **Behavior-preservation** — a refactoring step must not change API responses, DB schema, or observable behavior. Include an equivalence check in the AC (e.g. compare responses for the same input, existing tests pass).

### D. Generate files

Once the user approves, create the files below.

#### D-1. `phases/index.json` (top-level index)

Top-level index tracking multiple tasks. If it already exists, append to the `phases` array.

```json
{
  "phases": [
    {
      "dir": "0-supervisor-cleanup",
      "status": "pending"
    }
  ]
}
```

- `dir`: task directory name.
- `status`: `"pending"` | `"completed"` | `"error"` | `"blocked"`. Updated automatically by execute.py.
- Timestamps (`completed_at`, `failed_at`, `blocked_at`) are recorded automatically by execute.py on status change. Do not set them at creation time.

#### D-2. `phases/{task-name}/index.json` (task detail)

```json
{
  "project": "청년정책지원 챗봇 Backend",
  "phase": "<task-name>",
  "steps": [
    { "step": 0, "name": "step-slug", "status": "pending" },
    { "step": 1, "name": "step-slug", "status": "pending" }
  ]
}
```

Field rules:

- `project`: project name (see CLAUDE.md).
- `phase`: task name, matching the directory name.
- `steps[].step`: 0-indexed sequence number.
- `steps[].name`: kebab-case slug.
- `steps[].status`: all start as `"pending"`.

State transitions and auto-recorded fields:

| Transition | Fields recorded | Recorded by |
|------|-------------|----------|
| → `completed` | `completed_at`, `summary` | Claude session (summary), execute.py (timestamp) |
| → `error` | `failed_at`, `error_message` | Claude session (message), execute.py (timestamp) |
| → `blocked` | `blocked_at`, `blocked_reason` | Claude session (reason), execute.py (timestamp) |

`summary` is a one-line recap of the step's output, written on completion. execute.py accumulates it into subsequent steps' prompts as context — so make it useful for the next step (files created, key decisions, etc).

`created_at` is written once at task level on execute.py's first run. Step-level `started_at` is also written automatically by execute.py when each step begins. Don't set either at creation time.

#### D-3. `phases/{task-name}/step{N}.md` (one per step)

```markdown
# Step {N}: {name}

## Files to read

Read these files first to understand the project's architecture and design intent:

- `/docs/ARCHITECTURE.md`
- `/docs/ADR.md`
- {files created/modified by prior steps}
- {files targeted by this refactoring}

Read the code from prior steps carefully and understand the design intent before acting.

## Task

{Concrete implementation instructions: file paths, class/function signatures, logic description.
Code snippets should stay at the interface/signature level — leave the implementation to the agent.
But make any hard constraints explicit.
State whether this step is a refactor (behavior-preserving) or a bug fix (behavior-changing).}

## Acceptance Criteria

```bash
python -m pytest       # existing tests still pass
python -m mypy .        # no type errors
```

## Verification steps

1. Run the AC commands above.
2. Check the architecture checklist:
   - Does it follow ARCHITECTURE.md's directory structure and graph topology?
   - Does it stay within ADR's tech stack choices?
   - Does it respect CLAUDE.md's CRITICAL rules?
   - (For refactor steps) Are the API response shape / DB schema unchanged?
3. Update the corresponding step in `phases/{task-name}/index.json` based on the result:
   - Success → `"status": "completed"`, `"summary": "one-line recap of the output"`
   - Still failing after 3 fix attempts → `"status": "error"`, `"error_message": "specific error"`
   - Needs user intervention (API keys, auth, manual setup, etc.) → `"status": "blocked"`, `"blocked_reason": "specific reason"`, then stop immediately

## Do not

- {What not to do in this step. Format: "Don't do X. Reason: Y."}
- Don't break existing tests
- Don't add features not specified in this step
```

### E. Execute

```bash
python3 scripts/execute.py {task-name}        # run sequentially
python3 scripts/execute.py {task-name} --push  # run then push
```

What execute.py handles automatically:

- Creating/checking out branch `feat/{task-name}`
- Injecting guardrails — CLAUDE.md + docs/*.md content in every step's prompt
- Accumulating context — passing completed steps' summaries into subsequent prompts
- Self-correction — retrying up to 3 times on failure, feeding back the prior error message
- Two-stage commits — separate `Feat:` (code) and `Chore:` (metadata) commits, per this project's commit convention
- Timestamps — auto-recording started_at, completed_at, failed_at, blocked_at

`execute.py` runs each step as a new unattended session via `claude -p --dangerously-skip-permissions` and commits automatically. It acts without real-time human approval, so review the phase's step design thoroughly before running it for the first time.

Error recovery:

- **On error**: in `phases/{task-name}/index.json`, set the failed step's `status` back to `"pending"`, remove `error_message`, and re-run.
- **On blocked**: resolve the reason in `blocked_reason`, then set `status` to `"pending"`, remove `blocked_reason`, and re-run.
