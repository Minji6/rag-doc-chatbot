#!/bin/bash
# PreToolUse hook (matcher: Bash). Hard-blocks dangerous commands.
# Input: JSON on stdin, {"tool_input": {"command": "..."}, ...} (Claude Code hooks schema)
# Block: exit code 2 (stderr becomes the block reason shown to Claude)
#
# Parses JSON with python instead of jq — jq is not installed in this environment.

input=$(cat)
cmd=$(echo "$input" | python -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print(data.get('tool_input', {}).get('command', ''), end='')
except Exception:
    pass
")

[[ -z "$cmd" ]] && exit 0

if echo "$cmd" | grep -qE 'rm[[:space:]]+-rf|git[[:space:]]+push[[:space:]]+(--force|-f\b)|git[[:space:]]+reset[[:space:]]+--hard|git[[:space:]]+clean[[:space:]]+-f|DROP[[:space:]]+TABLE|TRUNCATE[[:space:]]+TABLE'; then
  echo "BLOCKED: dangerous command detected: ${cmd}" >&2
  exit 2
fi

exit 0
