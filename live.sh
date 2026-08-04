#!/usr/bin/env bash
# 브라우저 라이브 뷰 원커맨드 실행기 (REMOTE_VIEWING.md §3 의 간편판).
#
#   bash live.sh              # 빈 창고 씬만 3시간 스트리밍
#   bash live.sh level2       # level2 solution 미션을 라이브로
#   bash live.sh level1 base  # level1 base 미션을 라이브로
#   bash live.sh stop         # 라이브 씬 + HTTP 서버 모두 종료
#
# 시청: 노트북 브라우저에서 http://<서버IP>:49100
# 미션 라이브 런은 시각화 전용 — 끝나면 벤치마크 결과를 자동 원복한다.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${LIVE_PORT:-49100}"

if [[ "${1:-}" == "stop" ]]; then
  pkill -f live_mjpeg_sim.py 2>/dev/null && echo "[live] 씬 종료" || true
  pkill -f mjpeg_server.py 2>/dev/null && echo "[live] 서버 종료" || true
  exit 0
fi

# 이 호스트는 Isaac 동시 1런만 가능 — 미션 런이 돌고 있으면 건드리지 않는다
if pgrep -f "carter_run.py" > /dev/null; then
  echo "[live] 이미 미션 런이 실행 중 — 끝나기를 기다렸다 다시 실행하세요." >&2
  exit 1
fi
pkill -f live_mjpeg_sim.py 2>/dev/null && sleep 2 || true

# MJPEG HTTP 서버는 시뮬레이터와 독립 — 없을 때만 띄운다
if ! ss -tln 2>/dev/null | grep -q ":$PORT "; then
  nohup python3 levels/common/mjpeg_server.py "$PORT" > /dev/null 2>&1 &
  echo "[live] MJPEG 서버 시작 (port $PORT)"
fi

IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo "[live] 브라우저에서 열기: http://${IP:-<서버IP>}:$PORT"

export CARTER_RECORD=1 CARTER_RECORD_DIR=/dev/shm/carter_live
export CARTER_RECORD_EXT=jpg CARTER_RECORD_ROTATE=10
export CARTER_LIVE_WAIT="${CARTER_LIVE_WAIT:-60}"

if [[ -z "${1:-}" ]]; then
  echo "[live] 빈 창고 씬 스트리밍 (3시간, Ctrl-C로 종료)"
  exec bash levels/run_isaac.sh levels/common/live_mjpeg_sim.py
fi

LEVEL="$1"
VARIANT="${2:-solution}"
RUNNER="levels/$LEVEL/${VARIANT}_carter_run.py"
if [[ ! -f "$RUNNER" ]]; then
  echo "[live] 러너 없음: $RUNNER  (예: bash live.sh level2 / bash live.sh level1 base)" >&2
  exit 1
fi

echo "[live] $LEVEL $VARIANT 미션 라이브 — 부팅 ~1.5분 + 대기 ${CARTER_LIVE_WAIT}초 후 주행 시작"
bash levels/run_isaac.sh "$RUNNER"

# 라이브 런은 시각화 전용: 덮어쓴 벤치마크 결과(tracked 파일만)를 원복
git checkout -- "levels/$LEVEL/results/" 2>/dev/null || true
echo "[live] 미션 종료 — 벤치마크 결과 파일은 원복했음 (라이브 런은 채점에 안 씀)"
