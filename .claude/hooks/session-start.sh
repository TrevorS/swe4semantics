#!/bin/bash
set -euo pipefail

# Only run in Claude Code on the web environment
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
    exit 0
fi

# Change to project directory
cd "$CLAUDE_PROJECT_DIR"

# Configure git author
git config user.email "trevor@strieber.org"
git config user.name "Trevor Strieber"

# Install all dependencies including dev tools (ruff, ty, pytest)
# uv sync is idempotent and uses the cached container state efficiently
uv sync --dev

echo "Dependencies installed successfully"
