#!/bin/bash
# PostToolUse hook (matcher: Bash). Warns if the same command produces an
# "error-looking" result 5+ times within 60 seconds.
#
# Note: the Claude Code hooks docs don't specify exactly which fields PostToolUse
# exposes for exit code / stdout / stderr, so this uses a heuristic scan of
# tool_response text for error signals. False positives are possible, so this
# defaults to warn-only (exit 0).
#
# Parses JSON with python instead of jq — jq is not installed in this environment.

export PYTHONIOENCODING=utf-8

input=$(cat)

cmd=$(echo "$input" | python -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print(data.get('tool_input', {}).get('command', ''), end='')
except Exception:
    pass
")

response=$(echo "$input" | python -c "
import json, sys
try:
    data = json.load(sys.stdin)
    resp = data.get('tool_response', '')
    if not isinstance(resp, str):
        resp = json.dumps(resp, ensure_ascii=False)
    print(resp, end='')
except Exception:
    pass
")

[[ -z "$cmd" ]] && exit 0
echo "$response" | grep -qiE 'error|exception|traceback|failed' || exit 0

state_dir="${CLAUDE_PROJECT_DIR:-$(pwd)}/.claude/hooks"
mkdir -p "$state_dir"
state_file="$state_dir/.circuit-breaker-state.json"
[[ -f "$state_file" ]] || echo '{}' > "$state_file"

sig=$(echo -n "$cmd" | md5sum | cut -d' ' -f1)
now=$(date +%s)

python - "$state_file" "$sig" "$now" <<'PYEOF'
import json, sys

state_file, sig, now = sys.argv[1], sys.argv[2], int(sys.argv[3])
try:
    with open(state_file, encoding="utf-8") as f:
        state = json.load(f)
except Exception:
    state = {}

history = [t for t in state.get(sig, []) if now - t < 60]
history.append(now)
state[sig] = history

with open(state_file, "w", encoding="utf-8") as f:
    json.dump(state, f)

if len(history) >= 5:
    print(
        f"WARN: the same command produced an error-looking result {len(history)} times "
        f"within 60 seconds. Stop retrying the same approach and change strategy.",
        file=sys.stderr,
    )
PYEOF

exit 0
