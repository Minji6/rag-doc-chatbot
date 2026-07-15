#!/bin/bash
# PreToolUse hook (matcher: Edit|Write). Checks whether an api/ implementation file
# being edited has a corresponding test file.
#
# Defaults to warn-only (exit 0). This project has no tests/ directory yet, so a hard
# block (exit 2) would immediately block every edit under api/. Once a tests/ structure
# is in place, change the final `exit 0` to `exit 2` to enforce it.
#
# Parses JSON with python instead of jq — jq is not installed in this environment.

input=$(cat)
file_path=$(echo "$input" | python -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print(data.get('tool_input', {}).get('file_path', ''), end='')
except Exception:
    pass
")

[[ -z "$file_path" ]] && exit 0

# Normalize path separators (Windows backslashes)
norm_path="${file_path//\\//}"

case "$norm_path" in
  */api/*.py|api/*.py) ;;
  *) exit 0 ;;
esac

case "$(basename "$norm_path")" in
  test_*.py|*_test.py) exit 0 ;;
esac

project_dir="${CLAUDE_PROJECT_DIR:-$(pwd)}"
base=$(basename "$norm_path" .py)
rel_dir=$(dirname "$norm_path")
rel_dir="${rel_dir#*api/}"

candidates=(
  "$project_dir/tests/$rel_dir/test_${base}.py"
  "$project_dir/tests/test_${base}.py"
)

for t in "${candidates[@]}"; do
  if [[ -f "$t" ]]; then
    exit 0
  fi
done

echo "WARN: no test file found for ${file_path}. Consider writing tests/${rel_dir}/test_${base}.py first. (TDD Guard — warn-only for now)" >&2
exit 0
