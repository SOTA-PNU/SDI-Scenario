# ---- A-2-1 test-run telemetry (REPORT.md 실험 15) -----------------------------------
# agent_loop.py --a21 appends this block after [END EDIT REGION] in run_isaac_sim test
# runners only; graded runners never contain it. It changes no command and no physics:
#   * run_mission's controller argument is wrapped at the call; the module-level name
#     `controller`, its attributes and globals stay exactly the model's;
#   * env.raycast_scan on the harness instance is wrapped to note what the scans returned;
#   * chassis contacts, scored or not, are read with PhysX get_contact_report() at the
#     start of each controller call (a second contact subscriber would run inside the
#     physics step; polling moved runs no more than two plain runs of the same code differ)
#     and folded into episodes per touched prim on the spot;
#   * a controller exception is noted and re-raised unchanged.
# Everything goes to results/<variant>_a21diag.json when run_mission returns or raises.
_A21_WATCH = []  # agent_loop fills in the EDIT REGION's module-level names
_A21_CONTACTS = True


def _a21_install():
    import json as _json
    import os as _os
    import sys as _sys
    import time as _time
    import traceback as _tb
    from pathlib import Path as _Path

    from arena_harness import ArenaHarness as _AH

    _file = _Path(__file__).resolve()
    _variant = _file.stem.replace("_carter_run", "")
    _orig_run = _AH.run_mission
    _CH = "/World/Nova_Carter_ROS/chassis_link"
    _ROBOT = "/World/Nova_Carter_ROS"

    def _run(self, controller_fn, *args, **kwargs):
        rows, raw, episodes, open_ep = [], [], [], {}
        scans = {"calls": 0, "last": None, "args": {}}
        changes, prev, initial = [], {}, {}
        state = {"t": 0.0, "calls": 0, "pose": None}
        error = {}
        orig_scan = self.raycast_scan

        def scan(*a, **k):
            r = orig_scan(*a, **k)
            scans["calls"] += 1
            try:
                key = "raycast_scan(" + ", ".join([repr(x) for x in a]
                                                  + [f"{kk}={vv!r}" for kk, vv in sorted(k.items())]) + ")"
                if key not in scans["args"] and len(scans["args"]) < 6:
                    scans["args"][key] = round(state["t"], 3)
                rng = k.get("max_range", a[3] if len(a) > 3 else 6.0)
                m = min(r, key=lambda bd: bd[1]) if r else (0.0, rng)
                last = scans["last"]
                if last is None or last[0] != state["t"] or m[1] < last[1]:
                    scans["last"] = (state["t"], round(float(m[1]), 3), round(float(m[0]), 3))
            except Exception:  # noqa: BLE001
                pass
            return r

        self.raycast_scan = scan

        def simple(v):
            if isinstance(v, bool) or isinstance(v, int):
                return v
            if isinstance(v, str) and len(v) <= 16:
                return v
            return None

        def snapshot(fn):
            snap = {}
            g = getattr(fn, "__globals__", {})
            for name in _A21_WATCH:
                if name in g and not callable(g[name]):
                    v = simple(g[name])
                    if v is not None:
                        snap[name] = v
            for owner in [fn] + [g[n] for n in _A21_WATCH if callable(g.get(n)) and g.get(n) is not fn]:
                for an, av in list(getattr(owner, "__dict__", {}).items())[:20]:
                    v = simple(av)
                    if v is not None:
                        snap[f"{getattr(owner, '__name__', '?')}.{an}"] = v
            return snap

        poll = None
        try:
            if _A21_CONTACTS:
                import omni.physx
                from pxr import PhysicsSchemaTools

                _iface = omni.physx.get_physx_simulation_interface()
                _paths = {}

                def _p(i):
                    v = _paths.get(i)
                    if v is None:
                        v = _paths[i] = str(PhysicsSchemaTools.intToSdfPath(i))
                    return v

                def poll(now):
                    rep = _iface.get_contact_report()
                    headers, data = rep[0], rep[1]
                    for h_ in headers:
                        typ = str(h_.type).split(".")[-1]
                        if "LOST" in typ:
                            continue
                        a0, a1 = _p(h_.actor0), _p(h_.actor1)
                        # the harness's rule (carter_env.boot.on_contact) decides scored
                        harness_partner = a1 if a0.startswith(_CH) else a0
                        touched, rcol = (a1, _p(h_.collider0)) if a0.startswith(_ROBOT) else (a0, _p(h_.collider1))
                        if touched.startswith(_ROBOT):
                            continue
                        t = round(now, 3)
                        # episodes per touched prim, gaps of 0.5 s or less joined: O(1) per report
                        e = open_ep.get(touched)
                        if e is not None and t - e[3] <= 0.5:
                            e[3] = t
                            e[4] += 1
                            continue
                        pt, sep = None, None
                        if h_.num_contact_data:
                            cd = data[h_.contact_data_offset]
                            pt = [round(cd.position[0], 3), round(cd.position[1], 3), round(cd.position[2], 3)]
                            sep = round(float(cd.separation), 4)
                        e = [touched, harness_partner, t, t, 1, pt, sep, rcol]
                        open_ep[touched] = e
                        if len(episodes) < 5000:
                            episodes.append(e)
                        if len(raw) < 500:
                            raw.append([t, typ, harness_partner, touched, rcol, pt, sep])
        except Exception as e:  # noqa: BLE001
            poll = None
            raw.append([None, "contact_poll_unavailable", repr(e)[:160], None, None, None, None])

        def wrapped(t, pose, env):
            state["t"] = t
            state["pose"] = pose
            if poll is not None:
                try:
                    poll(t)
                except Exception:  # noqa: BLE001
                    pass
            try:
                out = controller_fn(t, pose, env)
            except Exception as e:  # noqa: BLE001
                try:
                    frames = _tb.extract_tb(_sys.exc_info()[2])
                    error.update(type=type(e).__name__, msg=str(e)[:300], t=round(t, 3),
                                 pose=[round(pose[0], 3), round(pose[1], 3), round(pose[2], 3)],
                                 calls_before=state["calls"],
                                 frames=[[_Path(f.filename).name, f.lineno, f.name, (f.line or "")[:160]]
                                         for f in frames][-12:])
                except Exception:  # noqa: BLE001
                    pass
                raise
            state["calls"] += 1
            try:
                if not (isinstance(out, (tuple, list)) and len(out) == 3):
                    raise TypeError("not a 3-tuple")   # left for the harness to judge, as in grading
                v, w, done = out
                last = scans["last"]
                hit = last is not None and last[0] == t
                rows.append((round(t, 3), round(pose[0], 4), round(pose[1], 4), round(pose[2], 4),
                             round(float(v), 4), round(float(w), 4), 1 if done else 0,
                             last[1] if hit else None, last[2] if hit else None))
            except Exception:  # noqa: BLE001
                rows.append((round(t, 3), round(pose[0], 4), round(pose[1], 4), round(pose[2], 4),
                             None, None, None, None, None))
            try:
                if state["calls"] == 1 or state["calls"] % 30 == 0:
                    snap = snapshot(controller_fn)
                    if state["calls"] == 1:
                        initial.update(snap)
                        prev.update(snap)
                    elif len(changes) < 400:
                        for name, val in snap.items():
                            if prev.get(name, "<unset>") != val:
                                changes.append((round(t, 3), name, prev.get(name), val,
                                                round(pose[0], 3), round(pose[1], 3)))
                                prev[name] = val
            except Exception:  # noqa: BLE001
                pass
            return out

        try:
            return _orig_run(self, wrapped, *args, **kwargs)
        finally:
            self.raycast_scan = orig_scan
            try:
                t0 = self._t0 or 0.0
                path = _os.path.join(str(_file.parent / "results"), f"{_variant}_a21diag.json")
                with open(path, "w") as f:
                    _json.dump({"variant": _variant, "written": _time.strftime("%F %T"),
                                "columns": ["t", "x", "y", "yaw", "v_cmd", "w_cmd", "done",
                                            "scan_min_m", "scan_min_bearing"],
                                "rows": rows,
                                "episode_columns": ["touched", "harness_partner", "t0", "t1", "reports",
                                                    "first_point", "first_separation", "robot_collider"],
                                "contact_episodes": episodes,
                                "contacts_raw": raw,
                                "harness_collisions": [(round(c[0] - t0, 3), str(c[1]).split(".")[-1], c[2])
                                                       for c in self.collisions[:2000]],
                                "scan_calls": scans["calls"], "runner": _file.name,
                                "scan_args": [[k, v] for k, v in scans["args"].items()],
                                "state_initial": initial, "state_changes": changes, "watched": list(_A21_WATCH),
                                "controller_error": error or None}, f, separators=(",", ":"))
            except Exception as e2:  # noqa: BLE001
                print(f"[a21] diag write failed: {e2!r}", flush=True)

    _AH.run_mission = _run


_a21_install()
# ---- end A-2-1 test-run telemetry -------------------------------------------------------
