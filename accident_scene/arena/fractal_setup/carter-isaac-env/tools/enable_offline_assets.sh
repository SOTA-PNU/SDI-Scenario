#!/usr/bin/env bash
# Point the harness at a local asset mirror instead of NVIDIA's S3 bucket.
#
#   bash tools/enable_offline_assets.sh /data/isaac-assets
#   bash tools/enable_offline_assets.sh --revert
#
# Only run this if the target server cannot reach the internet. The default
# online path is what the source host used and needs no patch at all.
#
# What it changes: Isaac resolves the scene through get_assets_root_path(),
# which reads the carb setting /persistent/isaac/asset_root/default. That
# setting can only be injected through SimulationApp's "extra_args" — Kit does
# not read sys.argv — so the three files that construct a SimulationApp get one
# extra element, guarded by $CARTER_ASSET_ROOT. With that variable unset the
# behaviour is identical to the unpatched harness.
set -uo pipefail

CARTER_REPO="${CARTER_REPO:-$HOME/carter-scenario-levels}"
TARGETS=(
  "$CARTER_REPO/levels/common/carter_env.py"
  "$CARTER_REPO/levels/common/probe_scene.py"
  "$CARTER_REPO/levels/common/live_mjpeg_sim.py"
)

if [ $# -lt 1 ]; then
  echo "usage: $0 <mirror-dir> | --revert" >&2
  exit 64
fi

python3 - "$1" "${TARGETS[@]}" <<'PY'
import os, re, sys

mode, files = sys.argv[1], sys.argv[2:]
MARK = "# [offline-assets]"
ORIG = '"extra_args": ["--/rtx/verifyDriverVersion/enabled=false"],'

for path in files:
    if not os.path.isfile(path):
        print(f"  skip (not found): {path}")
        continue
    src = open(path).read()

    if mode == "--revert":
        if MARK not in src:
            print(f"  clean already: {path}")
            continue
        # collapse the patched block back to the single original line
        src = re.sub(
            r'([ \t]*)"extra_args": \[.*?\]\s*\+\s*\(\[[^\]]*\]\s*if[^)]*\)\s*,\s*'
            + re.escape(MARK) + r'\n',
            lambda m: m.group(1) + ORIG + "\n",
            src, flags=re.S)
        open(path, "w").write(src)
        print(f"  reverted: {path}")
        continue

    if MARK in src:
        print(f"  already patched: {path}")
        continue
    if ORIG not in src:
        print(f"  WARNING: expected line not found, left untouched: {path}")
        continue
    if not re.search(r'^import os$', src, flags=re.M):
        src = re.sub(r'^(import [a-z])', r'import os\n\1', src, count=1, flags=re.M)

    def repl(m):
        ind = m.group(1)
        return (f'{ind}"extra_args": ["--/rtx/verifyDriverVersion/enabled=false"]\n'
                f'{ind}    + (["--/persistent/isaac/asset_root/default="\n'
                f'{ind}        + os.environ["CARTER_ASSET_ROOT"]]\n'
                f'{ind}       if os.environ.get("CARTER_ASSET_ROOT") else []),  {MARK}\n')

    src = re.sub(r'^([ \t]*)' + re.escape(ORIG) + r'\n', repl, src, count=1, flags=re.M)
    open(path, "w").write(src)
    print(f"  patched: {path}")
PY

if [ "$1" = "--revert" ]; then
  echo "Reverted. Remove CARTER_ASSET_ROOT from your environment too."
  exit 0
fi

MIRROR="$(cd "$1" 2>/dev/null && pwd)" || { echo "no such directory: $1" >&2; exit 66; }
for d in Isaac NVIDIA; do
  [ -d "$MIRROR/$d" ] || { echo "mirror is missing $MIRROR/$d — get_assets_root_path() stats both" >&2; exit 66; }
done
SCENE="$MIRROR/Isaac/Samples/ROS2/Scenario/carter_warehouse_navigation.usd"
[ -r "$SCENE" ] && echo "  scene present: $SCENE" \
                || echo "  WARNING: scene missing at $SCENE — run tools/mirror_assets.py first"

cat <<EOF

Done. Export this for every run (add it to \$CARTER_WS/env.sh to make it stick):

    export CARTER_ASSET_ROOT="$MIRROR"

Verify with a smoke test — the log should print '[env] opening $MIRROR/...'
instead of an https:// URL.
EOF
