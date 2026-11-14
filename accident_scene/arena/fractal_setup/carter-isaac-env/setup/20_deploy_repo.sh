#!/usr/bin/env bash
# Deploy the benchmark repo to $CARTER_REPO and freeze the reference results.
#
# Deploys from this package's payload/ by default. Set CARTER_USE_GIT=1 to clone
# from GitHub instead (needs access to a private repo; the payload is the same
# source tree at the same commit, minus ~190 MB of demo videos and a pptx).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

die() { bad "$1"; exit 1; }

PAYLOAD="$PKG_DIR/payload/carter-scenario-levels"
REPO_URL="https://github.com/hyunjun1234/carter-scenario-levels.git"
PINNED_COMMIT="$(cat "$PKG_DIR/payload/COMMIT" 2>/dev/null || echo unknown)"

step "1. Deploy $CARTER_REPO"
if [ -d "$CARTER_REPO/levels/common" ]; then
  ok "already deployed: $CARTER_REPO"
elif [ "${CARTER_USE_GIT:-0}" = "1" ]; then
  git clone "$REPO_URL" "$CARTER_REPO" || die "git clone failed (private repo — check your credentials, or unset CARTER_USE_GIT to use the bundled payload)"
  if [ "$PINNED_COMMIT" != "unknown" ]; then
    git -C "$CARTER_REPO" checkout "$PINNED_COMMIT" || warn "could not check out $PINNED_COMMIT; you are on whatever main points at now"
  fi
  ok "cloned at $(git -C "$CARTER_REPO" rev-parse --short HEAD)"
  warn "the git clone does NOT carry this package's portability patches: upstream still
         defaults CARTER_WS to /home/jun/carter_ws. Every script here exports CARTER_WS, so
         the benchmark is unaffected — but do not call levels/run_isaac.sh directly without
         exporting it first. See payload/carter-scenario-levels/PORTABILITY-PATCHES.md."
else
  [ -d "$PAYLOAD" ] || die "payload missing at $PAYLOAD"
  mkdir -p "$CARTER_REPO"
  cp -a "$PAYLOAD/." "$CARTER_REPO/" || die "copy failed"
  ok "deployed from payload (source commit $PINNED_COMMIT)"
fi

step "2. Make it a git checkout"
# Several helpers (live.sh, and the documented recovery path) restore the
# reference results with `git checkout -- levels/<n>/results/`. On a plain
# extracted tarball that silently no-ops and the baseline is lost on first run.
if [ -d "$CARTER_REPO/.git" ]; then
  ok "already a git repo"
else
  git -C "$CARTER_REPO" init -q 2>/dev/null \
    && git -C "$CARTER_REPO" add -A 2>/dev/null \
    && git -C "$CARTER_REPO" -c user.email=carter@local -c user.name=carter \
         commit -qm "reference state (source commit $PINNED_COMMIT)" 2>/dev/null \
    && ok "initialised a local git repo so 'git checkout -- levels/*/results/' works" \
    || warn "git init failed — reference restore via git will not work. The read-only snapshot in step 3 is then your only baseline."
fi

step "3. Freeze the reference results (read-only baseline)"
# Every run writes <variant>_result.json into the SAME results dir, overwriting
# the committed reference. Snapshot it before anything runs.
BASELINE="$CARTER_WS/reference_baseline"
if [ -d "$BASELINE" ]; then
  ok "baseline already frozen at $BASELINE"
else
  mkdir -p "$BASELINE"
  ( cd "$CARTER_REPO" && tar -cf - levels/*/results levels_zero2level/*/results 2>/dev/null ) \
    | tar -xf - -C "$BASELINE" 2>/dev/null
  N="$(find "$BASELINE" -name '*_result.json' 2>/dev/null | wc -l)"
  chmod -R a-w "$BASELINE" 2>/dev/null || true
  ok "froze $N reference result files at $BASELINE (read-only)"
fi

step "4. Install the run wrapper"
mkdir -p "$CARTER_WS/tools/bin"
install -m 755 "$PKG_DIR/tools/carter-run" "$CARTER_WS/tools/bin/carter-run" \
  && ok "installed $CARTER_WS/tools/bin/carter-run" \
  || die "could not install carter-run"

step "5. GPU-free sanity check of the deployed tree"
# These three checkers need no GPU and no Isaac. Run them inside the Isaac env,
# because check_yaml_sync.py needs PyYAML and a fresh host's system python
# usually has none.
cd "$CARTER_REPO" || die "cannot cd $CARTER_REPO"
RC=0
for lvl in 0 1 2 3; do
  isaac_py levels/common/check_edit_region.py "levels/level$lvl" >/dev/null 2>&1 || RC=1
done
[ $RC -eq 0 ] && ok "edit-region invariant holds for level0..3" || warn "check_edit_region.py reported a difference"
isaac_py levels/common/check_ladder.py    >/dev/null 2>&1 && ok "ladder invariant holds (L(n).base == L(n-1).solution)" || warn "check_ladder.py failed"
isaac_py levels/common/check_yaml_sync.py >/dev/null 2>&1 && ok "scenario YAML <-> runner constants in sync"         || warn "check_yaml_sync.py failed"

step "Done"
ok "repo at $CARTER_REPO, baseline at $CARTER_WS/reference_baseline"
say "    next: setup/30_smoke_test.sh"
