#!/usr/bin/env bash
# furio (openclaude + NPU router) launched the way ~/.local/bin/furio launches it, except that
# OPENAI_BASE_URL is left to the caller so the traffic can go through llm_log_proxy.py
# (the furio wrapper hard-sets it to :8400). REPORT.md 실험 14.
#   OPENAI_BASE_URL=http://127.0.0.1:8411/v1 OPENCLAUDE_CONFIG_DIR=<dir> bash furio_direct.sh <openclaude args>
set -euo pipefail
export CLAUDE_CODE_USE_OPENAI=1
: "${OPENAI_BASE_URL:?set OPENAI_BASE_URL (the logging proxy)}"
: "${OPENCLAUDE_CONFIG_DIR:?set OPENCLAUDE_CONFIG_DIR (a per-run config dir)}"
export OPENAI_BASE_URL OPENCLAUDE_CONFIG_DIR
export OPENAI_MODEL="${OPENAI_MODEL:-gpt-oss-120b}"
export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-8192}"
export OPENAI_API_KEY="$( [ -f "$HOME/.furio/key" ] && cat "$HOME/.furio/key" || echo dummy )"
if [ -z "${CLAUDE_CODE_OPENAI_CONTEXT_WINDOWS:-}" ] && [ -s "$HOME/.furio/ctx.json" ]; then
  export CLAUDE_CODE_OPENAI_CONTEXT_WINDOWS="$(cat "$HOME/.furio/ctx.json")"
fi
export API_TIMEOUT_MS="${API_TIMEOUT_MS:-900000}"
export CLAUDE_STREAM_IDLE_TIMEOUT_MS="${CLAUDE_STREAM_IDLE_TIMEOUT_MS:-600000}"
_BUN="$(command -v bun 2>/dev/null || true)"; [ -n "$_BUN" ] || _BUN="$HOME/.bun/bin/bun"
exec "$_BUN" --no-env-file --config=/dev/null "$HOME/.furio/lib/node_modules/@gitlawb/openclaude/dist/cli.mjs" "$@"
