"""Check the vectorised NumPy step against a direct loop translation of the JS.

The reference below follows ``updateParticles``/``applyBorderForce`` line by
line, but reads every neighbour from the start-of-step state (Jacobi order),
which is the semantics of the port.  All pairs are checked (no neighbour
list), so this also validates the KD-tree pair search.
"""
import math
import sys

import numpy as np

from lava_lamp import sph as L


def ref_border(p, poly, H, K):
    n = len(poly)
    area = sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1] for i in range(n))
    ccw = area > 0
    inside = False
    j = n - 1
    for i in range(n):
        vi, vj = poly[i], poly[j]
        if ((vi[1] > p["y"]) != (vj[1] > p["y"])) and (
                p["x"] < (vj[0] - vi[0]) * (p["y"] - vi[1]) / (vj[1] - vi[1]) + vi[0]):
            inside = not inside
        j = i
    if not inside:
        p0x, p0y = p["x"] - p["vx"], p["y"] - p["vy"]
        best, ix, iy = math.inf, None, None
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]
            s1x, s1y = p["x"] - p0x, p["y"] - p0y
            s2x, s2y = b[0] - a[0], b[1] - a[1]
            den = -s2x * s1y + s1x * s2y
            if den != 0:
                s = (-s1y * (p0x - a[0]) + s1x * (p0y - a[1])) / den
                t = (s2x * (p0y - a[1]) - s2y * (p0x - a[0])) / den
                if 0 <= s <= 1 and 0 <= t <= 1 and t < best:
                    best, ix, iy = t, p0x + t * s1x, p0y + t * s1y
        if ix is not None:
            p["x"], p["y"] = ix, iy
        else:
            md, sx, sy = math.inf, p["x"], p["y"]
            for i in range(n):
                a, b = poly[i], poly[(i + 1) % n]
                ex, ey = b[0] - a[0], b[1] - a[1]
                l2 = ex * ex + ey * ey
                t = max(0, min(1, ((p["x"] - a[0]) * ex + (p["y"] - a[1]) * ey) / l2))
                qx, qy = a[0] + t * ex, a[1] + t * ey
                d = math.hypot(p["x"] - qx, p["y"] - qy)
                if d < md:
                    md, sx, sy = d, qx, qy
            p["x"], p["y"] = sx, sy
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ex, ey = b[0] - a[0], b[1] - a[1]
        l2 = ex * ex + ey * ey
        t = max(0, min(1, ((p["x"] - a[0]) * ex + (p["y"] - a[1]) * ey) / l2))
        dx, dy = p["x"] - (a[0] + t * ex), p["y"] - (a[1] + t * ey)
        d = math.hypot(dx, dy)
        if d < H:
            q = d / H
            f = K * (1 - q) ** 2
            if p["mat"] == 1:
                f *= 1.25
            if d > 1e-7:
                nx, ny = dx / d, dy / d
            else:
                el = math.sqrt(l2)
                nx, ny = (-ey / el, ex / el) if ccw else (ey / el, -ex / el)
            p["vx"] += f * nx
            p["vy"] += f * ny


def ref_step(S, P: L.LavaParams, W=400.0, Hc=600.0, poly=None):
    poly = [tuple(v) for v in (L.lamp_polygon(W, Hc) if poly is None else poly)]
    n = S.shape[1]
    ps = [dict(zip(L.STATE_NAMES, S[:, i])) for i in range(n)]
    # stage 1: gravity, heater
    for p in ps:
        p["vy"] += P.gravity * p["m"]
        if p["mat"] == 1:
            rate = (0.002 - abs(p["x"] - W / 2) / W * 0.0015) * P.temp_speed
            goal = P.goal_hot if p["y"] > Hc - P.heater_height else p["m"]
            p["m"] += (goal - p["m"]) * rate
            if p["m"] < P.goal_cold:
                p["m"] += P.ambient_rate * rate
    pre = [dict(p) for p in ps]
    H, H2 = P.h, P.h * P.h
    out = []
    for i, p1 in enumerate(pre):
        q1 = dict(p1)
        for j, p2 in enumerate(pre):
            if j == i:
                continue
            dx, dy = p1["x"] - p2["x"], p1["y"] - p2["y"]
            d2 = dx * dx + dy * dy
            if not (0 < d2 < H2):
                continue
            dist = math.sqrt(d2)
            q = dist / H
            st = (2.5 / P.st_ratio) if p1["mat"] == 0 else P.st_ratio
            if p1["mat"] == p2["mat"] == 1 and j < i:
                q1["m"] += (p2["m"] - p1["m"]) * P.cond_rate
            if p1["mat"] != p2["mat"]:
                st = 0
            pr = P.stiffness * (1 - q) ** 2 - 0.04 * st
            q1["vx"] += dx / dist * pr
            q1["vy"] += dy / dist * pr
            rvx, rvy = p1["vx"] - p2["vx"], p1["vy"] - p2["vy"]
            dot = rvx * dx + rvy * dy
            visc = P.visc_wax if p1["mat"] == 1 else P.visc_water
            if p1["mat"] != p2["mat"]:
                visc = 0
            if dot > 0:
                imp = dot / (d2 + 0.01 * H2) * visc * (1 - q)
                q1["vx"] -= dx / dist * imp
                q1["vy"] -= dy / dist * imp
        sp = math.hypot(q1["vx"], q1["vy"])
        if sp > P.max_vel:
            q1["vx"] *= P.max_vel / sp
            q1["vy"] *= P.max_vel / sp
        q1["x"] += q1["vx"]
        q1["y"] += q1["vy"]
        ref_border(q1, poly, H, P.stiffness)
        out.append([q1[k] for k in L.STATE_NAMES])
    return np.array(out).T


def reference_errors(n_steps=5):
    """Max abs difference from the reference after each of ``n_steps`` steps."""
    rng = np.random.default_rng(1)
    n = 400
    S = L.initial_state(n, 0.3, seed=2)
    # scramble a bit: random velocities and temperatures, some near the heater,
    # and some particles outside the lamp to exercise the snapping code
    S[2:4] = rng.normal(0, 2.0, (2, n))
    wax = S[5] == 1
    S[4, wax] = rng.uniform(-6, 1.5, wax.sum())
    S[1, :40] = rng.uniform(585, 600, 40)
    S[0, 40:60] = rng.uniform(0, 400, 20)
    P = L.LavaParams(st_ratio=3.0, visc_wax=0.4)

    sim = L.LavaLampSim(state=S.copy(), params=P)
    S_ref = S.copy()
    errors = []
    for _ in range(n_steps):
        S_ref = ref_step(S_ref, P)
        sim.step()
        errors.append(np.max(np.abs(sim.state - S_ref)))
    return errors


def test_matches_reference():
    assert max(reference_errors()) < 1e-9


def main():
    errors = reference_errors()
    for k, err in enumerate(errors):
        print(f"step {k + 1}: max |numpy - reference| = {err:.3e}")
    ok = max(errors) < 1e-9
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
