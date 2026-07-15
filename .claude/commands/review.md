Review this project's pending changes.

First read:
- `/CLAUDE.md`
- `/docs/ARCHITECTURE.md`
- `/docs/ADR.md`

Then inspect the changed files and verify against this checklist:

## Checklist

1. **Architecture compliance**: Does it follow ARCHITECTURE.md's directory structure and the LangGraph graph topology (fan-out/fan-in)?
2. **Tech stack compliance**: Does it stay within ADR's technology choices?
3. **Test coverage**: Are new/changed logic paths covered by tests?
4. **CRITICAL rules**: Does it respect CLAUDE.md's CRITICAL rules (composer's sole responsibility, whitelist field-name matching, etc.)?
5. **Behavior preservation**: If the change is a refactor, are API responses and DB schema unchanged?
6. **Build/test passes**: Do `python -m pytest` and `python -m mypy .` pass without errors?

## Output format

| Item | Result | Notes |
|------|------|------|
| Architecture compliance | ✅/❌ | {detail} |
| Tech stack compliance | ✅/❌ | {detail} |
| Test coverage | ✅/❌ | {detail} |
| CRITICAL rules | ✅/❌ | {detail} |
| Behavior preservation | ✅/❌ | {detail} |
| Build/test passes | ✅/❌ | {detail} |

For any violation, propose a concrete fix.
