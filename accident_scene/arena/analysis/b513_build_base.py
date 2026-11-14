"""Build base_carter_run_ros_map_lidartime_objects.py (B-5-13, 실험 39) from the B-5-11/B-5-12 base
base_carter_run_ros_map_dbg_heapq_lidartime.py: (1) the tracking block with observed_objects() before the DBG block,
(2) _robot_only records the pose and updates the tracking every step before calling controller(), (3) the sensor base
code's internal names _S and _BODY become _BASE_SENSOR_STATE and _BASE_SELF_BOX (models often write _S and _BODY; the
B-5-13 review showed a redefined _BODY silently breaks the self filter). v2 after the review (wf_d6a4e0ca-94e): every-step
update, vel_ok and track_age_s, small fragments merged, stale points and lost tracks dropped every step, a copy kept for
the new-content test, neutral wording. The first version is results/archive_base_carter_run_ros_map_lidartime_objects_v1.py.

    python3 analysis/b513_build_base.py   # refuses to overwrite
"""
import re
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
OUT = ARENA / "base_carter_run_ros_map_lidartime_objects.py"
SRC = ARENA / "base_carter_run_ros_map_dbg_heapq_lidartime.py"

BLOCK = '''# ---- 감지 기본 코드 (물체 추적) ----
# 앞뒤 2D 라이다로 지도에 없는 물체를 묶어 추적합니다. base 가 controller() 를 부르기 직전마다 로봇 자세를 적고(_BASE_POSES)
# 추적을 갱신하므로(_base_track_step), controller() 안에서는 observed_objects() 로 결과만 받으면 됩니다.
# 이 부분도 [EDIT REGION] 밖이므로 고치지 마십시오.
_BASE_OBJ_CFG = dict(age=0.1, near=8.0, margin=0.25, voxel=0.1, gap=0.45, gate=1.0, vwin=0.8, vmin_n=4, vmin_span=0.3,
                     vmax=2.0, keep=1.0, stale=0.3, frag_n=3, frag_join=0.6)
_BASE_POSES = []                                  # (t, x, y, yaw), 최근 1 s
_BASE_OBJ_ST = {"prev": {"front": None, "back": None}, "pts": {"front": None, "back": None},
                "pts_t": {"front": None, "back": None}, "tracks": [], "nid": 0, "error": None}


def _base_obj_pose_at(t):
    """적어 둔 자세에서 시각 t 의 자세를 직선으로 보간합니다."""
    h = _BASE_POSES
    if not h:
        return None
    if t <= h[0][0]:
        return h[0][1:]
    for (ta, xa, ya, wa), (tb, xb, yb, wb) in zip(reversed(h[:-1]), reversed(h[1:])):
        if ta <= t:
            f = 0.0 if tb == ta else (t - ta) / (tb - ta)
            dw = math.atan2(math.sin(wb - wa), math.cos(wb - wa))
            return xa + f * (xb - xa), ya + f * (yb - ya), wa + f * dw
    return h[-1][1:]


def _base_obj_cluster(P):
    """gap 이하로 떨어진 점끼리 묶습니다."""
    n = len(P)
    if n == 0:
        return []
    d = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1]) <= _BASE_OBJ_CFG["gap"]
    label = -np.ones(n, dtype=int)
    groups = []
    for i in range(n):
        if label[i] >= 0:
            continue
        label[i] = len(groups)
        stack, members = [i], []
        while stack:
            k = stack.pop()
            members.append(k)
            for m in np.nonzero(d[k] & (label < 0))[0]:
                label[m] = label[i]
                stack.append(m)
        groups.append(P[members])
    return groups


def _base_obj_velocity(hist):
    """최근 vwin 초의 중심 위치를 직선으로 맞춘 속도 (vx, vy, 맞춤 여부). 자료가 모자라면 (0, 0, False)."""
    cfg = _BASE_OBJ_CFG
    t_end = hist[-1][0]
    h = np.array([q for q in hist if t_end - q[0] <= cfg["vwin"]])
    if len(h) < cfg["vmin_n"] or h[-1, 0] - h[0, 0] < cfg["vmin_span"]:
        return 0.0, 0.0, False
    tt = h[:, 0] - h[:, 0].mean()
    den = float((tt * tt).sum())
    vx = float((tt * (h[:, 1] - h[:, 1].mean())).sum() / den)
    vy = float((tt * (h[:, 2] - h[:, 2].mean())).sum() / den)
    sp = math.hypot(vx, vy)
    if sp > cfg["vmax"]:
        vx, vy = vx * cfg["vmax"] / sp, vy * cfg["vmax"] / sp
    return vx, vy, True


def _base_track_step(t):
    """base 가 controller() 를 부르기 직전마다 부릅니다. 새 스캔이 왔으면(내용이 직전과 다르면) 그 스캔을 잰 때의 자세로
    점을 바꿔 물체를 다시 묶고 추적을 이어 붙입니다. 1.0 s 넘게 보이지 않은 물체는 매 스텝 지웁니다."""
    st, cfg = _BASE_OBJ_ST, _BASE_OBJ_CFG
    t_scan = t - cfg["age"]
    new = False
    for which in ("front", "back"):
        p = lidar2d_points(which)
        q = st["prev"][which]
        if q is not None and p.shape == q.shape and np.array_equal(p, q):
            continue
        st["prev"][which] = np.array(p, copy=True)
        pose = _base_obj_pose_at(t_scan)
        st["pts"][which] = to_world(p, pose)[:, :2] if len(p) and pose is not None else np.zeros((0, 2))
        st["pts_t"][which] = t_scan
        new = True
    tracks = st["tracks"]
    if new and _BASE_POSES:
        x0, y0 = _BASE_POSES[-1][1], _BASE_POSES[-1][2]
        parts = [P for w, P in st["pts"].items()
                 if P is not None and len(P) and t_scan - st["pts_t"][w] <= cfg["stale"]]
        P = np.concatenate(parts) if parts else np.zeros((0, 2))
        if len(P):
            P = P[np.hypot(P[:, 0] - x0, P[:, 1] - y0) < cfg["near"]]
        if len(P):
            P = unmapped_points(P, cfg["margin"])
        if len(P):
            _, keep = np.unique(np.floor(P / cfg["voxel"]).astype(np.int64), axis=0, return_index=True)
            P = P[keep]
        groups = _base_obj_cluster(P)
        big = [g for g in groups if len(g) >= cfg["frag_n"]]
        for g in (g for g in groups if len(g) < cfg["frag_n"]):
            if big:
                dmin = [float(np.hypot(B[:, None, 0] - g[None, :, 0], B[:, None, 1] - g[None, :, 1]).min()) for B in big]
                k = int(np.argmin(dmin))
                if dmin[k] <= cfg["frag_join"]:
                    big[k] = np.concatenate([big[k], g])
                    continue
            big.append(g)
        for tr in tracks:
            tr["hit"] = False
        for grp in big:
            c = grp.mean(axis=0)
            best, bd = None, cfg["gate"]
            for tr in tracks:
                if tr["hit"]:
                    continue
                lt, lx, ly = tr["hist"][-1]
                dd = math.hypot(c[0] - (lx + tr["vx"] * (t_scan - lt)), c[1] - (ly + tr["vy"] * (t_scan - lt)))
                if dd < bd:
                    best, bd = tr, dd
            if best is None:
                best = {"id": st["nid"], "hist": [], "vx": 0.0, "vy": 0.0, "vel_ok": False, "born_t": t_scan}
                st["nid"] += 1
                tracks.append(best)
            best["hit"], best["seen_t"] = True, t_scan
            best["radius"] = float(np.hypot(grp[:, 0] - c[0], grp[:, 1] - c[1]).max())
            best["n_points"] = int(len(grp))
            best["hist"] = [h for h in best["hist"] if t_scan - h[0] <= 2.0] + [(t_scan, float(c[0]), float(c[1]))]
            best["vx"], best["vy"], best["vel_ok"] = _base_obj_velocity(best["hist"])
    st["tracks"] = [tr for tr in tracks if t_scan - tr["seen_t"] <= cfg["keep"]]


def observed_objects():
    """지도에 없는 물체를 추적한 결과를 리스트로 돌려줍니다. 가까운 것부터 정렬되어 있습니다.

    base 가 controller() 를 부르기 직전마다 추적을 갱신하므로, controller() 안에서 언제 몇 번 불러도 같은 결과를 줍니다.
    갱신: 앞뒤 2D 라이다에 새 스캔이 오면(내용이 직전과 다를 때만), 그 스캔을 잰 때(받기 0.1 s 전)의 로봇 자세로 점을
    월드 좌표로 바꿉니다. 로봇에서 8 m 안이면서 지도 물체와 벽에서 0.25 m 넘게 떨어진 점만 남겨 0.1 m 칸으로 솎고, 0.45 m
    이하로 떨어진 점끼리 묶어 물체 하나로 봅니다(점이 3개보다 적은 조각은 0.6 m 안의 큰 묶음에 합칩니다). 묶음은 직전 물체의
    예측 위치에서 1.0 m 안이면 같은 물체로 이어 붙이고, 최근 0.8 s 의 중심 위치를 직선으로 맞춰 속도를 구합니다.

    물체마다 dict: id(같은 물체면 같은 번호), x, y(보인 점들의 중심, 월드 좌표 [m]), vx, vy, speed([m/s]),
    vel_ok(속도를 맞출 자료가 충분했는지: 최근 0.8 s 안에 4번 넘게, 0.3 s 넘게 보였을 때 True), radius(중심에서 가장 먼
    점까지 [m]), n_points(솎은 뒤 점 수), seen_t(그 물체를 마지막으로 본 스캔의 시각 [s]), age_s(지금 시각 - seen_t),
    track_age_s(처음 본 스캔부터 seen_t 까지 [s]), dist(지금 pose 의 점에서 중심까지 [m]).
    x, y, vx, vy 는 seen_t 때의 값입니다. 1.0 s 넘게 보이지 않은 물체는 빠집니다.

    주의: 라이다는 로봇 쪽을 향한 면만 보므로 중심은 물체의 실제 중심과 다르고, 보이는 면이 바뀌면 중심이 옮겨 가 속도에
    잡음이 섞입니다. 속도는 최근 0.8 s 위치의 직선 맞춤이라 실제 변화보다 늦게 따라갑니다. vel_ok 가 False 면 vx, vy 는
    0 이지만 정지했다는 뜻이 아닙니다. 지도에 없는 정지 물체도 나옵니다."""
    if not _BASE_POSES:
        return []
    t_now, x0, y0, _ = _BASE_POSES[-1]
    out = []
    for tr in _BASE_OBJ_ST["tracks"]:
        lt, lx, ly = tr["hist"][-1]
        out.append({"id": tr["id"], "x": lx, "y": ly, "vx": tr["vx"], "vy": tr["vy"], "speed": math.hypot(tr["vx"], tr["vy"]),
                    "vel_ok": tr["vel_ok"], "radius": tr["radius"], "n_points": tr["n_points"], "seen_t": lt,
                    "age_s": t_now - lt, "track_age_s": lt - tr["born_t"], "dist": math.hypot(lx - x0, ly - y0)})
    out.sort(key=lambda o: o["dist"])
    return out


'''

WRAP_OLD = '''def _robot_only(t, pose, env):
    """The controller sees no harness handle (env is None): only the robot's own sensors."""
    _BASE_DBG_STATE["t"] = t
    return controller(t, pose, None)
'''
WRAP_NEW = '''def _robot_only(t, pose, env):
    """The controller sees no harness handle (env is None): only the robot's own sensors."""
    _BASE_DBG_STATE["t"] = t
    _BASE_POSES.append((t, pose[0], pose[1], pose[2]))       # observed_objects(): poses of the last 1 s
    while _BASE_POSES and t - _BASE_POSES[0][0] > 1.0:
        _BASE_POSES.pop(0)
    try:
        _base_track_step(t)                                  # tracking updated every step, before the controller
    except Exception as e:                                   # noqa: BLE001  a tracking fault must not stop the run
        if _BASE_OBJ_ST["error"] is None:
            _BASE_OBJ_ST["error"] = repr(e)
            print("DBG [base] observed_objects() 갱신 오류:", repr(e))
    return controller(t, pose, None)
'''


def build(src):
    a = src.index("# ---- Carter 센서 기본 코드 ----")
    b = src.index("# ---- 지도 기본 코드 ----")
    sensor = src[a:b]
    sensor = re.sub(r"\b_S\b", "_BASE_SENSOR_STATE", sensor)
    sensor = sensor.replace("_BODY = (-0.70, 0.25, -0.34, 0.34)                                     # 로봇 자신의 몸체 범위",
                            "_BASE_SELF_BOX = (-0.70, 0.25, -0.34, 0.34)                            # 자기 점을 지우는 상자(몸체보다 큼)")
    sensor = re.sub(r"\b_BODY\b", "_BASE_SELF_BOX", sensor)
    out = src[:a] + sensor + src[b:]
    anchor = "# ---- DBG 기록 ----"
    assert out.count(anchor) == 1 and out.count(WRAP_OLD) == 1
    out = out.replace(anchor, BLOCK + anchor, 1).replace(WRAP_OLD, WRAP_NEW, 1)
    assert "_BODY" not in out and not re.search(r"\b_S\b", out)
    return out


if __name__ == "__main__":
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; not overwriting")
    OUT.write_text(build(SRC.read_text()))
    s = OUT.read_text()
    print("wrote", OUT.name, "| lines", s.count("\n"), "| EDIT REGION marker line", s[:s.index("# [EDIT REGION] mission")].count("\n") + 1)
