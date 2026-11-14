"""Analysis-only outer chooser adapter; never installed in an operational base.

Old controller/helpers remain in their original namespace. Only unverified raw
returns go through the separately loaded v8 chooser, using original observed
surface points. Same-tick chooser results may be reused, or checked again using
their ORIGINAL request/target (feeding the chosen pair back is not idempotent).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from b514_build_base import archive_before_write
from b515_check_offline import load_ns, safe_json

ARENA = Path(__file__).resolve().parents[1]
BASE = ARENA / "base_carter_run_ros_map_lidartime_objects_check_v8.py"
OUT = ARENA / "results/analysis/b519_shield_unit.json"


def _digest(value):
    def convert(obj):
        if hasattr(obj, "tolist"): return obj.tolist()
        raise TypeError(f"Unsupported observation value: {type(obj).__name__}")
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), default=convert)
    return hashlib.sha256(data.encode()).hexdigest()


class OuterShield:
    """CPU adapter: step(t, pose) -> (v, w, done, meta).

    The caller updates the original pose history and tracker before step, as
    _robot_only does. This object neither advances sensors nor invents points.
    strategy='none' is the unmodified command/done baseline with instrumentation.
    guard_done=False preserves the model's done exactly for isolated command
    comparisons. guard_done=True additionally requires zero command and MISSION
    goal tolerances. Neither mode claims actual velocity has reached zero.
    """

    def __init__(self, original_ns, outer_ns=None, strategy="ahead", verified_policy="reuse", guard_done=True):
        if strategy not in ("none", "ahead", "goal"):
            raise ValueError("strategy must be none, ahead or goal")
        if verified_policy not in ("reuse", "replay_original"):
            raise ValueError("verified_policy must be reuse or replay_original")
        self.ns = original_ns
        self.outer = load_ns(BASE) if outer_ns is None else outer_ns
        if self.outer is self.ns:
            raise ValueError("The outer v8 namespace must be independent")
        self.strategy, self.verified_policy, self.guard_done = strategy, verified_policy, bool(guard_done)
        self.original_choose = self.ns.get("choose_command")
        self.records, self.last_meta = [], None
        self._active, self._tick = False, 0
        self._frame_objects, self._frame_context = [], None
        self.stats = {k: 0 for k in ("ticks", "inner_calls", "verified_ticks", "unverified_ticks", "outer_calls",
                                   "changed_commands", "done_suppressed", "replayed_inner_calls", "baseline_ticks")}
        # The map is controller-visible static data. Use the same map for both namespaces.
        if "WALLS" in self.ns: self.outer["WALLS"] = tuple(self.ns["WALLS"])
        if "map_rects" in self.ns:
            self.outer["_MAP_RECTS"][:] = [self.ns["map_rects"]().copy()]
        self.outer["observed_objects"] = lambda: copy.deepcopy(self._frame_objects)
        if self.original_choose is not None:
            self._wrapper = self._record_choose
            self.ns["choose_command"] = self._wrapper

    def close(self):
        if self.original_choose is not None and self.ns.get("choose_command") is self._wrapper:
            self.ns["choose_command"] = self.original_choose

    def _objects(self):
        provider = self.ns.get("observed_objects")
        if provider is None:
            raise ValueError("Original namespace has no observed_objects provider")
        return copy.deepcopy(provider())

    def _context(self):
        poses = self.ns.get("_BASE_POSES")
        current = tuple(float(q) for q in poses[-1]) if poses else None
        return current, _digest(self._objects())

    def _record_choose(self, *args, **kwargs):
        before = self._context() if self._active else None
        result = self.original_choose(*args, **kwargs)
        if self._active:
            self.stats["inner_calls"] += 1
            self.records.append({"tick": self._tick, "context": before,
                                 "args": copy.deepcopy(args), "kwargs": copy.deepcopy(kwargs),
                                 "pair": (float(result[0]), float(result[1])),
                                 "info": copy.deepcopy(result[2])})
        return result

    def _target(self, pose, pair):
        if self.strategy == "goal":
            goal = self.ns.get("GOAL", self.ns["MISSION"]["goal_xy_yaw"])
            return float(goal[0]), float(goal[1])
        # A simple heading point, not a reconstructed path or curved-command endpoint.
        distance = max(1.0, 3.0 * abs(pair[0]))
        direction = -1.0 if pair[0] < 0.0 else 1.0
        return (pose[0] + direction * distance * math.cos(pose[2]),
                pose[1] + direction * distance * math.sin(pose[2]))

    def _goal_ok(self, pose):
        mission = self.ns["MISSION"]
        goal = self.ns.get("GOAL", mission["goal_xy_yaw"])
        distance = math.hypot(pose[0] - goal[0], pose[1] - goal[1])
        yaw_error = abs(math.atan2(math.sin(pose[2] - goal[2]), math.cos(pose[2] - goal[2])))
        yaw_tol = mission.get("yaw_tolerance_rad")
        return distance <= mission["position_tolerance_m"] and (yaw_tol is None or yaw_error <= yaw_tol)

    def step(self, t, pose, controller=None, env=None):
        if self._active: raise RuntimeError("Nested shield steps are unsupported")
        t, pose = float(t), tuple(float(q) for q in pose)
        if len(pose) != 3 or not all(math.isfinite(q) for q in (t, *pose)):
            raise ValueError("Shield requires a finite actual time and pose")
        actual_pose_time = (t, *pose)
        if not self.ns.get("_BASE_POSES") or tuple(self.ns["_BASE_POSES"][-1]) != actual_pose_time:
            raise ValueError("Update the original pose history/tracker before shield.step")
        self._tick += 1
        self.stats["ticks"] += 1
        self.records = []
        self._frame_objects = self._objects()
        self._frame_context = (actual_pose_time, _digest(self._frame_objects))
        self._active = True
        try:
            raw = (self.ns["controller"] if controller is None else controller)(t, pose, None)
        finally:
            self._active = False
        raw_pair, raw_done = (float(raw[0]), float(raw[1])), bool(raw[2])
        if not all(math.isfinite(q) for q in raw_pair):
            raise ValueError("Controller returned a nonfinite command")
        current_context = self._context()
        eligible = [r for r in self.records if r["tick"] == self._tick
                    and r["context"] == self._frame_context == current_context and r["pair"] == raw_pair]
        record = eligible[-1] if eligible else None
        verified = record is not None
        self.stats["verified_ticks" if verified else "unverified_ticks"] += 1
        pair, info, target = raw_pair, copy.deepcopy(record["info"]) if verified else {}, None
        if self.strategy == "none":
            self.stats["baseline_ticks"] += 1
            route = "baseline_verified" if verified else "baseline_raw"
        elif verified:
            route = "verified_reuse"
            if self.verified_policy == "replay_original":
                replay = self.original_choose(*record["args"], **record["kwargs"])
                self.stats["replayed_inner_calls"] += 1
                if tuple(replay[:2]) != raw_pair:
                    raise RuntimeError("Same-context original-request replay changed the chosen pair")
                pair, info, route = tuple(replay[:2]), copy.deepcopy(replay[2]), "verified_original_request_replay"
        else:
            # Freeze the original sensor snapshot taken before the model controller.
            # Missing/empty points are not replaced with radii, centers or truth boxes.
            for obj in self._frame_objects:
                if "points" not in obj or not len(obj["points"]):
                    raise ValueError("Outer v8 requires actual nonempty observed point arrays")
            self.outer["_BASE_POSES"][:] = [actual_pose_time]
            target = self._target(pose, raw_pair)
            v, w, info = self.outer["choose_command"](*raw_pair, target)
            pair, info, route = (v, w), copy.deepcopy(info), "raw_v8_filter"
            self.stats["outer_calls"] += 1
        changed = pair != raw_pair
        self.stats["changed_commands"] += int(changed)
        done = raw_done
        if self.strategy != "none" and self.guard_done:
            done = raw_done and pair == (0.0, 0.0) and self._goal_ok(pose)
        self.stats["done_suppressed"] += int(raw_done and not done)
        info.setdefault("chosen", "raw")
        info["shield"] = {"strategy": self.strategy, "route": route, "verified": verified,
                          "tick": self._tick, "actual_pose_time": actual_pose_time,
                          "sensor_epoch_sha256": self._frame_context[1],
                          "raw_command": raw_pair, "output_command": pair, "changed": changed,
                          "fallback_target": target, "original_done": raw_done, "output_done": done,
                          "guard_done": self.guard_done, "inner_calls_this_tick": len(self.records),
                          "trusted_original_args": record["args"] if verified else None,
                          "trusted_original_kwargs": record["kwargs"] if verified else None}
        self.last_meta = info
        return pair[0], pair[1], done, info


def selftest():
    from b515_check_offline import object_fixture, seed
    from b519_check_offline import space
    checks, examples = {}, {}
    def record(name, ok, **details): checks[name] = {"pass": bool(ok), **details}
    def make(objects=(), strategy="ahead", verified_policy="reuse", guard_done=True):
        ns = space(objects)
        ns["GOAL"] = (5., 0., 0.)
        ns["MISSION"] = {**ns["MISSION"], "goal_xy_yaw": ns["GOAL"]}
        return ns, OuterShield(ns, strategy=strategy, verified_policy=verified_policy, guard_done=guard_done)
    def filtered(ns, args=(.3, 0., (5., 0.))):
        def controller(t, pose, env):
            v, w, _ = ns["choose_command"](*args)
            return v, w, False
        return controller
    ns, shield = make()
    result = shield.step(10., (0., 0., 0.), filtered(ns))
    record("same_tick_verified_pair_preserved_without_second_call", result[:2] == (1., 0.)
           and result[3]["shield"]["verified"] and shield.stats["outer_calls"] == 0)
    original = shield.original_choose; shield.close()
    record("close_restores_original_function", ns["choose_command"] is original)
    ns, shield = make(verified_policy="replay_original")
    result = shield.step(10., (0., 0., 0.), filtered(ns))
    record("literal_recheck_of_original_args_preserves_pair", result[:2] == (1., 0.)
           and shield.stats["replayed_inner_calls"] == 1 and shield.stats["outer_calls"] == 0)

    # Deterministic real-chooser counterexample: refeeding selectedfast changes it toslow.
    objects = [object_fixture(-.38492575641242466, .06655785537023018, -.2596065660481335, .49672357908356557)]
    args = (.3, 0., (2.8605613938573864, .8182005467460618))
    raw_ns = space(objects)
    first = raw_ns["choose_command"](*args)
    second = raw_ns["choose_command"](*first[:2], args[2])
    record("naive_double_choose_is_not_idempotent", first[:2] == (1., 0.) and second[:2] == (.5, 0.),
           original_request=args, first_pair=first[:2], second_pair=second[:2], first_reason=first[2]["reason"], second_reason=second[2]["reason"])
    for policy in ("reuse", "replay_original"):
        ns, shield = make(objects, verified_policy=policy)
        result = shield.step(10., (0., 0., 0.), filtered(ns, args))
        record("nonidempotent_case_preserved_"+policy, result[:2] == first[:2] and result[3]["shield"]["verified"])
        examples[policy] = result[3]

    ns, shield = make()
    first = shield.step(10., (0., 0., 0.), filtered(ns))
    # Same numeric pair and even same timestamp in a NEW controller call is not proof.
    second = shield.step(10., (0., 0., 0.), lambda *a: (*first[:2], False))
    record("numeric_pair_without_current_call_not_trusted", not second[3]["shield"]["verified"] and shield.stats["outer_calls"] == 1)
    def changed(t, pose, env):
        ns["choose_command"](.3, 0., (5., 0.))
        return .31, .1, False
    result = shield.step(10., (0., 0., 0.), changed)
    record("altered_returned_pair_filtered", not result[3]["shield"]["verified"] and shield.stats["outer_calls"] == 2)

    ns, shield = make()
    def wrong_pose(t, pose, env):
        ns["_BASE_POSES"][:] = [(t, 1., 0., 0.)]
        v, w, _ = ns["choose_command"](.3, 0., (5., 0.))
        ns["_BASE_POSES"][:] = [(t, *pose)]
        return v, w, False
    result = shield.step(10., (0., 0., 0.), wrong_pose)
    record("wrong_pose_call_not_trusted", not result[3]["shield"]["verified"] and shield.stats["outer_calls"] == 1)
    ns, shield = make([object_fixture(1.3, 1.2, 0., -.4)])
    def changed_sensor(t, pose, env):
        actual = ns["observed_objects"]
        ns["observed_objects"] = lambda: []
        v, w, _ = ns["choose_command"](.3, 0., (5., 0.))
        ns["observed_objects"] = actual
        return v, w, False
    result = shield.step(10., (0., 0., 0.), changed_sensor)
    record("different_sensor_epoch_not_trusted", not result[3]["shield"]["verified"]
           and shield.stats["outer_calls"] == 1 and len(shield._frame_objects) == 1)

    ns, shield = make()
    no_chooser = dict(ns); no_chooser.pop("choose_command")
    raw_shield = OuterShield(no_chooser, strategy="ahead", guard_done=False)
    result = raw_shield.step(10., (0., 0., 0.), lambda *a: (.3, 0., False))
    record("old_namespace_without_choose_supported", result[3]["shield"]["route"] == "raw_v8_filter")
    record("forward_heading_target", raw_shield._target((2., 3., math.pi/2), (.3, .4)) == (2., 4.))
    target = raw_shield._target((2., 3., math.pi/2), (-.6, .4))
    record("backward_heading_target", math.dist(target, (2., 1.2)) < 1e-12)
    raw_shield.strategy = "goal"
    record("goal_target_uses_only_visible_mission_goal", raw_shield._target((2., 3., 1.), (.3, .4)) == (5., 0.))
    ns, shield = make(strategy="none", guard_done=True)
    result = shield.step(10., (0., 0., 0.), lambda *a: (.27, -.14, True))
    record("none_mode_preserves_original_command_and_done", result[:3] == (.27, -.14, True) and shield.stats["outer_calls"] == 0)
    ns, shield = make(guard_done=False)
    result = shield.step(10., (0., 0., 0.), lambda *a: (.3, 0., True))
    record("primary_done_unchanged_even_when_not_zero_or_at_goal", result[2] is True and not shield.stats["done_suppressed"])
    ns, shield = make(guard_done=True)
    result = shield.step(10., (0., 0., 0.), lambda *a: (0., 0., True))
    record("optional_guard_rejects_false_done", result[2] is False and shield.stats["done_suppressed"] == 1)
    seed(ns, pose=(5., 0., 0.), now=11.)
    result = shield.step(11., (5., 0., 0.), lambda *a: (0., 0., True))
    record("optional_guard_accepts_goal_zero_command", result[:3] == (0., 0., True))
    ns, shield = make()
    ns["observed_objects"] = lambda: [{"id": 1, "x": 1., "y": 0., "vx": 0., "vy": 0., "vel_ok": True, "seen_t": 10.}]
    rejected = False
    try: shield.step(10., (0., 0., 0.), lambda *a: (.3, 0., False))
    except ValueError as exc: rejected = "point arrays" in str(exc)
    record("missing_points_rejected_without_fabrication", rejected)
    ns, shield = make()
    rejected = 0
    for pair in ((math.nan, 0.), (0., math.inf)):
        try: shield.step(10., (0., 0., 0.), lambda *a, pair=pair: (*pair, False))
        except ValueError: rejected += 1
    record("nonfinite_commands_rejected", rejected == 2)
    passed = sum(c["pass"] for c in checks.values())
    report = {"verdict": "pass" if passed == len(checks) else "fail", "passed": passed, "total": len(checks),
              "base_sha16": hashlib.sha256(BASE.read_bytes()).hexdigest()[:16], "checks": checks, "examples": examples,
              "primary_trial_contract": {"strategies": ["none", "ahead", "goal"], "guard_done": False,
                                         "verified_policy": "reuse", "outer_only_for_unverified_returns": True},
              "limitations": ["Analysis adapter only; no operational base or wrapper was changed.",
                              "Same-tick exact returned-pair/pose/observation evidence is semantic reuse, not an adversarial Python sandbox.",
                              "Old chooser versions are preserved for verified commands; the shield does not silently upgrade their policy to v8.",
                              "Raw fallbacks lack the intended waypoint. Ahead is a straight yaw point, including turning requests, not a curved-command endpoint.",
                              "Literal extra checking is supported only by replaying original inputs/target; refeeding selected pairs can change them.",
                              "Primary comparison preserves original done; runner must grade final pose using MISSION separately.",
                              "Observed point arrays are required; centers/radii/truth geometry never replace missing points."]}
    archive_before_write(OUT)
    OUT.write_text(json.dumps(safe_json(report), ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(f"{report['verdict']} {passed}/{len(checks)} {OUT}")
    for name, check in checks.items():
        if not check["pass"]: print("FAIL", name, check)
    return report


if __name__ == "__main__":
    raise SystemExit(0 if selftest()["verdict"] == "pass" else 1)
