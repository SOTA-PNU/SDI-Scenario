#!/usr/bin/env bash
# claude_sandbox.sh RUN_DIR [claude args...]
#
# Run Claude Code with nothing of this machine visible except a fresh home.
# Inside, RUN_DIR/home is mounted as /home/jun (cwd = /home/jun/workspace) and
# RUN_DIR/tmp as /tmp, so everything the agent writes - scratch simulators
# included - survives for the record. /usr and /etc are read-only; the project
# tree, ~/.claude (memory, transcripts, history.jsonl, plugins, user settings)
# and every other home file are absent, so the only task information the agent
# has is what the prompt carries - the same as the NPU entry (REPORT.md, 실험 7).
#
# Auth is the logged-in account's current OAuth *access* token, passed as
# CLAUDE_CODE_OAUTH_TOKEN. The refresh token never enters the sandbox, so the
# child cannot rotate the parent's login; once the access token expires the
# next launch reads the refreshed one from the credentials file.
set -euo pipefail
RUN_DIR="$(readlink -f "$1")"; shift
SBX_HOME="$RUN_DIR/home"
SBX_TMP="$RUN_DIR/tmp"
CLAUDE_BIN="$(readlink -f "$HOME/.local/bin/claude")"
# a long-lived token from `claude setup-token`, if the user saved one, never expires
# mid-run; otherwise fall back to the login's current access token
TOKEN_FILE="${CLAUDE_EQ_TOKEN_FILE:-$HOME/.config/claude_eq/oauth_token}"
if [ -s "$TOKEN_FILE" ]; then
  TOKEN="$(tr -d '[:space:]' < "$TOKEN_FILE")"
else
  TOKEN="$(python3 -c 'import json,os;print(json.load(open(os.path.expanduser("~/.claude/.credentials.json")))["claudeAiOauth"]["accessToken"])')"
fi
mkdir -p "$SBX_HOME/workspace" "$SBX_HOME/.local/bin" "$SBX_TMP"
[ -e "$SBX_HOME/.local/bin/claude" ] || : > "$SBX_HOME/.local/bin/claude"

exec env -i HOME=/home/jun USER=jun LOGNAME=jun LANG=C.UTF-8 TERM=dumb \
  PATH=/home/jun/.local/bin:/usr/local/bin:/usr/bin:/bin \
  CLAUDE_CODE_OAUTH_TOKEN="$TOKEN" DISABLE_AUTOUPDATER=1 \
  bwrap --die-with-parent --unshare-pid --unshare-ipc --unshare-uts \
    --ro-bind /usr /usr --symlink usr/bin /bin --symlink usr/sbin /sbin \
    --symlink usr/lib /lib --symlink usr/lib64 /lib64 \
    --ro-bind /etc /etc --ro-bind /run/systemd/resolve /run/systemd/resolve \
    --proc /proc --dev /dev --ro-bind /sys /sys \
    --bind "$SBX_HOME" /home/jun --bind "$SBX_TMP" /tmp \
    --ro-bind "$CLAUDE_BIN" /home/jun/.local/bin/claude \
    --chdir /home/jun/workspace \
    /home/jun/.local/bin/claude "$@"
