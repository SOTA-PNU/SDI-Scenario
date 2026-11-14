#!/usr/bin/env bash
# Stop Kit from ever creating a viewport, so the RTX hydra engine is never
# instantiated.
#
#   bash tools/enable_no_viewport.sh
#   bash tools/enable_no_viewport.sh --revert
#
# WHY THIS EXISTS
# ---------------
# On a host whose NVIDIA driver is from the R590 branch (595.x) — and on some
# R610 hosts — Kit 106.5 segfaults inside librtx.scenedb.plugin.so a
# millisecond after the RTX SceneDbContext is built. Measured on this host:
# every Kit log terminates on the identical "SceneDbContext : TLAS instance
# limit" line, and the line immediately before it is always the creation of
# HydraTexture 'omni_kit_widget_viewport_ViewportTexture_0'.
#
# The RTX *extension* is harmless: it only REGISTERS carb.scenerenderer-rtx and
# rtx.scenedb. They are INITIALIZED ~1.8 s later, when omni.kit.widget.viewport
# builds the viewport texture. Suppress the viewport window and the renderer
# never initialises — Kit boots, physics runs, OmniGraph ticks, ROS 2 works.
#
# WHAT IT DOES NOT CHANGE
# -----------------------
# sim.step(render=True) is left exactly as it is. It reduces to a single
# self._app.update(), which still ticks omni.physx.stageupdate and OmniGraph,
# so the ROS2SubscribeTwist -> DifferentialController -> ArticulationController
# chain and the PhysX contact reports are untouched. Every compared metric is
# read from physics state (get_world_pose, PhysX contact reports, PhysX
# raycasts), never from a rendered image, so parity is preserved.
#
# It also disables omni.physx.ui right after the ROS 2 bridge is enabled.
# That extension registers an unconditional stage-update callback doing
# ViewportCameraState(), which raises RuntimeError("No default or provided
# Viewport") on every tick once there is no viewport. Kit catches it but never
# finishes shutdown: measured 11.3M traceback lines and a 20-minute hang after
# the script had already printed its last line. It cannot be kept out with
# /app/extensions/excluded alone, because isaacsim.ros2.bridge pulls it back in
# through omni.isaac.dynamic_control and isaacsim.core.simulation_manager, and
# the exclusion list is not consulted by enable_extension(). It is UI-only;
# omni.physx itself (simulation, contact reports) is untouched.
#
# Finally it passes wait_for_replicator=False to SimulationApp.close().
# close() otherwise calls omni.replicator.core orchestrator.wait_until_complete(),
# which never returns when there are no render products: measured, the process
# spun on 12 cores inside omni.fabric.endframe for 11 minutes after the script
# had printed its last line. Teardown only — every result JSON is written before
# close() is reached, so no compared metric can move.
#
# WHAT IT COSTS
# -------------
# CARTER_RECORD, CARTER_RECORD_DIR, CARTER_LIVESTREAM and live_mjpeg_sim.py
# stop working: they need a render product. On an affected host they could not
# work anyway — the process dies before they run.
set -uo pipefail

CARTER_REPO="${CARTER_REPO:-$HOME/carter-scenario-levels}"
TARGETS=(
  "$CARTER_REPO/levels/common/carter_env.py"
  "$CARTER_REPO/levels/common/probe_scene.py"
  "$CARTER_REPO/levels/common/live_mjpeg_sim.py"
)

MODE="${1:-apply}"

python3 - "$MODE" "${TARGETS[@]}" <<'PY'
import os, re, sys

mode, files = sys.argv[1], sys.argv[2:]
MARK = "# [no-viewport]"
ORIG = '"extra_args": ["--/rtx/verifyDriverVersion/enabled=false"],'

# Verified on this host, in this order:
#   disableWindowOnLoad / entryCount=0 -> omni.kit.viewport.window never builds
#     a ViewportWindow, so no ViewportTexture and no HydraTexture.
#   autoManage/enabledList= (EMPTY, not "none") -> extension.py:60-63 does
#     `settings.get(...) or ""` then `if not managed_hd_exts: return`, so only
#     the empty string disables auto-manage. "none" is truthy and would let it
#     rewrite /renderer/enabled.
#   /renderer/enabled=none -> backstop. Must NOT be empty: texture.py:554 reads
#     an empty list as "any enabled renderer is allowed" and would pick rtx.
#   /renderer/active is deliberately left alone — setting it to "none" sends
#     texture.py:574 down set_extension_enabled_immediate("omni.hydra.none").
NEW = [
    '"--/rtx/verifyDriverVersion/enabled=false"',
    '"--/exts/omni.kit.viewport.window/startup/disableWindowOnLoad=true"',
    '"--/exts/omni.kit.viewport.window/windowMenu/entryCount=0"',
    '"--/exts/omni.kit.viewport.menubar.render/autoManage/enabledList="',
    '"--/renderer/enabled=none"',
    '"--/rtx/hydra/mdlMaterialWarmup=false"',
    # omni.physx.ui registers an unconditional stage-update callback that does
    # ViewportCameraState() -> RuntimeError("No default or provided Viewport")
    # once there is no viewport. Kit catches it but re-raises it every tick and
    # never finishes shutdown: measured 11.3M traceback lines and a 20-minute
    # hang after the script had already printed its last line. It is a UI-only
    # extension; omni.physx itself (simulation, contact reports) is untouched.
    # /app/extensions/excluded is the only lever that works —
    # /app/extensions/disableStartup segfaults (measured, rc=139).
    '"--/app/extensions/excluded/0=omni.physx.ui"',
]

MARK3 = "# [no-viewport-close]"
# [ \t]* not \s*: under re.M, \s* before $ swallows the final newline of a file
CLOSE_RE = re.compile(r'^([ \t]*)((?:self\.app|sim_app|app)\.close)\(\)[ \t]*$', re.M)


def patch_close(src):
    if MARK3 in src:
        return src
    return CLOSE_RE.sub(
        lambda m: f'{m.group(1)}{m.group(2)}(wait_for_replicator=False)  {MARK3}', src)


def unpatch_close(src):
    # [ \t]* not \s*: \s would also eat the newline after the marker, which on
    # the last line of a file silently drops the trailing newline
    return re.sub(r'\(wait_for_replicator=False\)[ \t]*' + re.escape(MARK3), '()', src)


BRIDGE = 'enable_extension("isaacsim.ros2.bridge")'
MARK2 = "# [no-viewport-physxui]"


def add_physxui_block(src):
    """Disable omni.physx.ui right after the ROS 2 bridge comes up."""
    if MARK2 in src or BRIDGE not in src:
        return src
    m = re.search(r'^([ \t]*)' + re.escape(BRIDGE) + r'\n([ \t]*)([A-Za-z_.]+)\.update\(\)\n',
                  src, flags=re.M)
    if not m:
        return src
    ind, app = m.group(2), m.group(3)
    # every emitted line carries MARK2 - drop_physxui_block removes by marker,
    # so a backslash-continued line without it would be left behind on --revert
    block = (f'{ind}import omni.kit.app as _kit_app  {MARK2}\n'
             f'{ind}_kit_app.get_app().get_extension_manager().set_extension_enabled_immediate('
             f'"omni.physx.ui", False)  {MARK2}\n'
             f'{ind}{app}.update()  {MARK2}\n')
    return src[:m.end()] + block + src[m.end():]


def drop_physxui_block(src):
    return "".join(l for l in src.splitlines(keepends=True) if MARK2 not in l)


for path in files:
    if not os.path.isfile(path):
        print(f"  skip (not found): {path}")
        continue
    src = open(path).read()

    if mode == "--revert":
        if MARK not in src and MARK2 not in src and MARK3 not in src:
            print(f"  clean already: {path}")
            continue
        src = drop_physxui_block(src)
        src = unpatch_close(src)
        src = re.sub(
            r'([ \t]*)"extra_args": \[[^\]]*\],\s*' + re.escape(MARK) + r'\n',
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

    def repl(m):
        ind = m.group(1)
        body = ("\n" + ind + "    ").join(a + "," for a in NEW)
        return f'{ind}"extra_args": [\n{ind}    {body}\n{ind}],  {MARK}\n'

    src = re.sub(r'^([ \t]*)' + re.escape(ORIG) + r'\n', repl, src, count=1, flags=re.M)
    src = add_physxui_block(src)
    src = patch_close(src)
    open(path, "w").write(src)
    print(f"  patched: {path}")
PY

if [ "$MODE" = "--revert" ]; then
  echo "Reverted to the stock harness."
else
  cat <<'EOF'

Done. Verify with:
    bash setup/30_smoke_test.sh
The Kit log should contain no 'SceneDbContext' and no 'HydraTexture' line.
EOF
fi
