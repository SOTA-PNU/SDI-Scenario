#!/usr/bin/env bash
# Check this host can run Isaac Sim 4.5 BEFORE downloading 7 GB of wheels.
# Exit 0 = every hard requirement met. Exit 1 = at least one blocker.
# Safe to re-run; changes nothing.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

FAILED=0
blocker() { bad "$1"; FAILED=$((FAILED + 1)); }

step "1. OS and glibc"
if [ -r /etc/os-release ]; then
  . /etc/os-release
  say "    distro : ${PRETTY_NAME:-unknown}"
fi
say "    kernel : $(uname -r)"
GLIBC="$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$')"
if [ -z "$GLIBC" ]; then
  warn "could not detect glibc version — verify manually that it is >= 2.34"
else
  # Isaac Sim 4.5 wheels are manylinux_2_34 -> glibc >= 2.34 is a hard floor.
  if awk -v v="$GLIBC" 'BEGIN{split(v,a,"."); exit !(a[1]>2 || (a[1]==2 && a[2]>=34))}'; then
    ok "glibc $GLIBC (>= 2.34 required by the manylinux_2_34 wheels)"
  else
    blocker "glibc $GLIBC is too old. Isaac Sim 4.5 wheels need >= 2.34 (Ubuntu 22.04+). Ubuntu 20.04 cannot install them at all."
  fi
fi

step "2. Disk space"
TARGET_DIR="$(dirname "$CARTER_WS")"
[ -d "$TARGET_DIR" ] || TARGET_DIR="$HOME"
AVAIL_GB="$(df -BG --output=avail "$TARGET_DIR" 2>/dev/null | tail -1 | tr -dc '0-9')"
if [ -n "$AVAIL_GB" ] && [ "$AVAIL_GB" -ge 30 ]; then
  ok "${AVAIL_GB} GB free on $TARGET_DIR (need ~25 GB: 15 GB env + 7 GB wheel cache)"
elif [ -n "$AVAIL_GB" ]; then
  blocker "only ${AVAIL_GB} GB free on $TARGET_DIR — need ~30 GB. Use --no-cache-dir during install to hold peak near 15 GB, or pick another CARTER_WS."
else
  warn "could not read free space for $TARGET_DIR"
fi

step "3. NVIDIA driver"
if ! have nvidia-smi; then
  blocker "nvidia-smi not found. Isaac Sim's gpu.foundation plugin shells out to /usr/bin/nvidia-smi at boot; without it Kit will not start."
else
  DRV="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
  GPUS="$(nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null)"
  say "    driver : ${DRV:-unknown}"
  say "    GPUs   :"
  printf '%s\n' "$GPUS" | sed 's/^/             /'
  if [ -n "$DRV" ]; then
    # Requirement is >= 535.129. Compare on major.minor only.
    if awk -v v="$DRV" 'BEGIN{split(v,a,"."); n=a[1]*1000+a[2]; exit !(n>=535129 || a[1]>535)}'; then
      ok "driver $DRV meets the >= 535.129 floor"
    else
      blocker "driver $DRV is below the 535.129 floor required by Isaac Sim 4.5."
    fi
  fi
fi

step "4. Vulkan / GL userspace  (the #1 cause of a dead Isaac Sim)"
# Kit's RTX renderer is Vulkan-only and finds the GPU ONLY through the ICD json.
# A compute-only driver (nvidia-headless-*, *-server in datacenter mode, or a
# container without NVIDIA_DRIVER_CAPABILITIES=graphics) has no ICD and Isaac
# Sim enumerates zero devices — reported as a confusing "no GPU" error.
ICD=""
for p in /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json; do
  [ -r "$p" ] && ICD="$p" && break
done
if [ -n "$ICD" ]; then
  ok "Vulkan ICD present: $ICD"
else
  blocker "no nvidia_icd.json under /usr/share/vulkan/icd.d or /etc/vulkan/icd.d. The driver was installed WITHOUT its GL/Vulkan userspace (libnvidia-gl-*). Install the full driver, not nvidia-headless-*. In Docker: --gpus all -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility"
fi

GLX="$(ls /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 2>/dev/null | head -1)"
if [ -n "$GLX" ]; then
  ok "libGLX_nvidia.so.0 present"
  MISSING_LIBS="$(ldd "$GLX" 2>/dev/null | grep 'not found' | awk '{print $1}' | tr '\n' ' ')"
  if [ -n "$MISSING_LIBS" ]; then
    blocker "libGLX_nvidia.so.0 has unresolved deps: $MISSING_LIBS  (install them; libX11/libXext are mandatory even fully headless with no DISPLAY)"
  else
    ok "libGLX_nvidia.so.0 dependencies all resolve"
  fi
else
  blocker "libGLX_nvidia.so.0 not found — same cause as a missing Vulkan ICD (driver installed without GL userspace)."
fi

step "5. System libraries the env does NOT bundle"
# The micromamba env carries its own libstdc++/libpython but zero X11/GL libs.
# Snapshot the ld.so cache once: `ldconfig -p | grep -q` SIGPIPEs ldconfig on an
# early match, and pipefail then misreports the hit as a miss.
LDCACHE="$( { ldconfig -p || /sbin/ldconfig -p || /usr/sbin/ldconfig -p; } 2>/dev/null )"
MISS=""
if [ -z "$LDCACHE" ]; then
  warn "could not read the ld.so cache — skipping this check. Verify manually that libX11, libvulkan and libGL are installed."
else
  for lib in libX11.so.6 libXext.so.6 libxcb.so.1 libXau.so.6 libXdmcp.so.6 \
             libvulkan.so.1 libGL.so.1 libGLdispatch.so.0 libGLX.so.0; do
    # bash pattern match, not a pipe: `... | grep -q` SIGPIPEs the producer on
    # an early match and pipefail then reports the hit as a miss.
    case "$LDCACHE" in
      *"$lib"*) ;;
      *) MISS="$MISS $lib" ;;
    esac
  done
fi
if [ -n "$LDCACHE" ] && [ -z "$MISS" ]; then
  ok "all required X11/GL/Vulkan client libraries present"
elif [ -n "$MISS" ]; then
  blocker "missing system libraries:$MISS
         sudo apt-get install -y libvulkan1 libglvnd0 libgl1 libglx0 libegl1 \\
              libx11-6 libxext6 libxcb1 libxau6 libxdmcp6 libxrandr2 libxi6 \\
              libxkbcommon0 libsm6 libice6"
fi

step "6. Outbound network to the Isaac asset server"
# get_assets_root_path() stats BOTH prefixes before returning; it raises
# RuntimeError if either fails, and there is no offline fallback.
NET_OK=1
S3_HOST="https://omniverse-content-production.s3-us-west-2.amazonaws.com"
SCENE_PATH="Assets/Isaac/4.5/Isaac/Samples/ROS2/Scenario/carter_warehouse_navigation.usd"

# omni.client resolves a prefix with an S3 LIST, not a GET. A plain GET on a
# prefix is a 404 even when the bucket is perfectly reachable, so probe with
# list-type=2 — the same operation check_server() ends up making.
for prefix in Isaac NVIDIA; do
  CODE="$(curl -s -o /dev/null -w '%{http_code}' --max-time 25 \
          "$S3_HOST/?list-type=2&prefix=Assets/Isaac/4.5/$prefix/&max-keys=1" 2>/dev/null)"
  if [ "$CODE" = "200" ]; then
    ok "S3 listable: Assets/Isaac/4.5/$prefix/"
  else
    NET_OK=0
    bad "S3 list failed for Assets/Isaac/4.5/$prefix/ (HTTP ${CODE:-timeout})"
  fi
done

CODE="$(curl -s -o /dev/null -w '%{http_code}' --max-time 25 "$S3_HOST/$SCENE_PATH" 2>/dev/null)"
if [ "$CODE" = "200" ]; then
  ok "scene USD fetchable over HTTPS:443"
else
  NET_OK=0
  bad "scene USD not fetchable over HTTPS:443 (HTTP ${CODE:-timeout})"
fi

# Kit switches object GETs to CloudFront over plain HTTP port 80.
CF="$(curl -s -o /dev/null -w '%{http_code}' --max-time 25 \
      "http://dcb18d6mfegct.cloudfront.net/$SCENE_PATH" 2>/dev/null)"
if [ "$CF" = "200" ]; then
  ok "CloudFront asset fetch reachable over HTTP:80"
else
  NET_OK=0
  warn "CloudFront (dcb18d6mfegct.cloudfront.net, port 80) returned ${CF:-timeout}. Kit uses plain HTTP:80 for object GETs — a firewall that only opens 443 to the S3 host is NOT sufficient."
fi

if [ "$NET_OK" -eq 0 ]; then
  blocker "the scene USD streams from the internet on EVERY run. Fix the network, or mirror the assets locally with tools/mirror_assets.py and set CARTER_ASSET_ROOT (see README section 'Offline / restricted network')."
fi

step "7. Environment hygiene"
if [ -n "${ROS_DISTRO:-}" ] && [ -z "${CARTER_ALLOW_ROS:-}" ]; then
  warn "ROS_DISTRO=$ROS_DISTRO is already set in this shell. Do NOT source a system ROS or RoboStack before running the benchmark — mixing that Humble ABI with the bridge's bundled one breaks the ROS 2 bridge silently. Start from a clean shell."
fi
if [ -d "$HOME/.local/lib/python3.10/site-packages" ]; then
  N="$(ls -1 "$HOME/.local/lib/python3.10/site-packages" 2>/dev/null | wc -l)"
  warn "~/.local/lib/python3.10/site-packages exists ($N entries). It would shadow the Isaac env; every script in this package sets PYTHONNOUSERSITE=1 to neutralise it. Do not run the benchmark without that guard."
else
  ok "no ~/.local/lib/python3.10 user site to shadow the env"
fi

step "Result"
if [ "$FAILED" -eq 0 ]; then
  ok "preflight passed — proceed with setup/10_install_env.sh"
  exit 0
else
  bad "$FAILED blocker(s). Fix them before installing; none of them get better later."
  exit 1
fi
