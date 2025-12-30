#!/bin/bash
set -euo pipefail

# Read hook input from stdin
input=$(cat)

# Extract tool name and file path
tool_name=$(echo "$input" | jq -r '.tool_name // empty')
file_path=$(echo "$input" | jq -r '.tool_input.file_path // empty')

# Only run for Edit or Write tools
if [[ "$tool_name" != "Edit" && "$tool_name" != "Write" ]]; then
    exit 0
fi

# Only run for Python files
if [[ ! "$file_path" =~ \.py$ ]]; then
    exit 0
fi

# Check if file exists
if [[ ! -f "$file_path" ]]; then
    exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

# Run ruff check (auto-fix safe issues)
echo "Running ruff check on $file_path..."
if ! uv run ruff check "$file_path" 2>&1; then
    echo "Ruff found issues in $file_path"
fi

# Run ty type check
echo "Running ty check on $file_path..."
if ! uv run ty check "$file_path" 2>&1; then
    echo "Type errors found in $file_path"
fi
