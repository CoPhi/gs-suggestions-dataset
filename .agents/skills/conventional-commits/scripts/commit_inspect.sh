#!/usr/bin/env bash
set -euo pipefail

echo "=== 1. Git Working Tree Status ==="
git status --short

echo ""
echo "=== 2. Diff Summary (Tracked Files) ==="
git diff --stat

echo ""
echo "=== 3. Staged Changes Summary ==="
git diff --cached --stat

echo ""
echo "=== 4. Recent Commit History (Style Reference) ==="
git log -n 5 --oneline
