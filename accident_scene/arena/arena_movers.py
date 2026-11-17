"""Moving obstacles for the project C real test (B-series prompts, REPORT.md 실험 19-).

Off unless ARENA_MOVERS names a scenario. The model's run_isaac_sim test drives never set
it, so the model develops on the fixed-obstacle map; only the final grading sets it.

Each mover is a kinematic box that walks back and forth on a straight segment p0 <-> p1 at a
constant speed, as a function of the mission clock t (the same t the controller receives),
so a scenario is the same in every run. Before the mission starts a mover sits at its t = 0
position. A mover never yields: it moves through whatever is in its way, and any contact
with the robot chassis is a scored collision (its prim is not under an excluded prefix).
"""

from __future__ import annotations

import math
import os
import random

SCENARIOS = {
    # 0.6 x 0.6 x 1.0 m boxes, taller than both 2D lidar planes (0.42 m) and the scan plane (0.35 m).
    # Phases put four of them on the A-4 seed route (npu_gpt_oss_120b_pA4_a0) when it passes
    # (t = 4.6, 26.1, 53.9, 78.7 s; analysis/c_movers_design.py), so a controller that does not look fails.
    "v1": [
        dict(name="start_zone", p0=(-7.0, -11.5), p1=(-7.0, -8.6), speed=0.5, phase=8.47, size=(0.6, 0.6, 1.0)),
        dict(name="west_corridor", p0=(-7.6, -1.0), p1=(-3.4, -1.0), speed=0.6, phase=6.08, size=(0.6, 0.6, 1.0)),
        dict(name="center", p0=(-3.4, 8.6), p1=(0.6, 8.6), speed=0.6, phase=4.9, size=(0.6, 0.6, 1.0)),
        dict(name="goal_west", p0=(4.0, 14.4), p1=(4.0, 17.3), speed=0.5, phase=6.46, size=(0.6, 0.6, 1.0)),
        dict(name="goal_south", p0=(4.7, 11.0), p1=(8.6, 11.0), speed=0.6, phase=0.0, size=(0.6, 0.6, 1.0)),
    ],
}


# Randomised scenarios (2026-11-15): the lanes of a base scenario, with each mover's speed and its start position on
# the lane (phase) drawn from a seeded RNG, so no controller can know where a mover will be from the run clock alone.
# ARENA_MOVERS_SEED fixes the draw (reproduce a run); unset, every run draws a fresh seed. The seed and the drawn
# values are printed by the harness and kept in results/<variant>_movers.json.
RANDOMIZED = {
    "v1r": {"lanes_of": "v1", "speed_range": (0.3, 0.8)},
}
_DRAWN = {}     # one draw per process: spawn and every physics step must see the same movers


def seed_used():
    return _DRAWN.get("seed")


def _randomized(name):
    if name in _DRAWN.get("by_name", {}):
        return [dict(m) for m in _DRAWN["by_name"][name]]
    spec = RANDOMIZED[name]
    env_seed = os.environ.get("ARENA_MOVERS_SEED", "").strip()
    seed = int(env_seed) if env_seed else random.SystemRandom().randrange(1, 2 ** 31)
    rng = random.Random(seed)
    lo, hi = spec["speed_range"]
    out = []
    for m in SCENARIOS[spec["lanes_of"]]:
        m = dict(m)
        (x0, y0), (x1, y1) = m["p0"], m["p1"]
        L = math.hypot(x1 - x0, y1 - y0)
        m["speed"] = round(rng.uniform(lo, hi), 4)
        m["phase"] = round(rng.uniform(0.0, 2.0 * L / m["speed"]), 4)     # anywhere on the lane, either direction
        out.append(m)
    _DRAWN["seed"] = seed
    _DRAWN.setdefault("by_name", {})[name] = out
    return [dict(m) for m in out]


def scenario_name():
    return os.environ.get("ARENA_MOVERS", "").strip() or None


def scenario(name=None):
    """The movers of a scenario; [] when ARENA_MOVERS is unset."""
    name = name or scenario_name()
    if not name:
        return []
    if name in RANDOMIZED:
        return _randomized(name)
    if name not in SCENARIOS:
        raise ValueError(f"unknown ARENA_MOVERS scenario {name!r}; known: {sorted(SCENARIOS) + sorted(RANDOMIZED)}")
    return [dict(m) for m in SCENARIOS[name]]


def position(m, t):
    """(x, y) of mover m at mission time t: a triangle wave p0 -> p1 -> p0 at m['speed']."""
    (x0, y0), (x1, y1) = m["p0"], m["p1"]
    L = math.hypot(x1 - x0, y1 - y0)
    s = (m["speed"] * (max(t, 0.0) + m["phase"])) % (2.0 * L)
    d = s if s <= L else 2.0 * L - s
    return x0 + (x1 - x0) * d / L, y0 + (y1 - y0) * d / L
