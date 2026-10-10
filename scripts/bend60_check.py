"""Achievable single-segment bend vs. a 60 deg Y-tube branch (hanging, gravity).

Design from sim_disc_design (2 x 4 discs, 30 mm gaps, PLA disc mass, tendons
routed through disc holes), E from the base config. One segment is commanded
to +/-60 deg (constant-curvature chord tendon lengths), the other straight.
Alignment metric: tip tangent angle from the trunk (mount) axis vs 60 deg.
Then searches the overdrive command that lands the tip axis at 60 deg and
checks it against actuator ctrlrange and force limits.

    uv run python scripts/bend60_check.py [--render] [--discs 4 --gap 30]
"""

import argparse
import json
import math

import numpy as np

import sim_disc_design as sd
import sweep_2seg as sw

DISCS, GAP, LPP = 4, 30.0, 3
TARGET = 60.0


def cc_ctrl(m, d, ctrl0, L, thetas, discs=None):
    """sd.cc_ctrl, but tolerant of a straight (theta = 0) segment and theta < 0."""
    discs = discs or DISCS
    u = ctrl0.copy()
    for a in range(m.nu):
        t = m.actuator_trnid[a, 0]
        seg = int(m.tendon(t).name.split("_")[1])
        x = sw.to_mount(m, d, d.site_xpos[m.wrap_objid[m.tendon_adr[t]]])[0]
        dl = 0.0
        for th in thetas[: seg + 1]:
            if abs(th) > 1e-9:
                R = L / th
                dl += discs * 2 * (R - x) * math.sin(th / discs / 2) - L
        u[a] = ctrl0[a] + dl
    return u


def run(xml, L, thetas):
    m, d = sd.build(xml, DISCS, LPP, True)
    ctrl0 = d.ctrl.copy()
    target = cc_ctrl(m, d, ctrl0, L, thetas, DISCS)
    max_force = [0.0]

    def ctrl_fn(s):
        max_force[0] = max(max_force[0], float(-d.actuator_force.min()))
        return ctrl0 + s * (target - ctrl0)

    qv = sw.settle(m, d, ctrl_fn, sw.SETTLE_MAX, tol=sw.SETTLE_TOL)
    ang = np.array([math.degrees(d.qpos[m.jnt_qposadr[j]]) * s for j, s in sw.bend_joints(m, d)])
    segs = np.array_split(ang, 2)
    _, Rm = sw.mount_frame(m, d)
    z = Rm.T @ d.xmat[m.body("EE_pos").id].reshape(3, 3)[:, 2]
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    margin = np.minimum(target - lo, hi - target)  # m left before ctrlrange clamps
    return dict(
        seg_bend=(float(segs[0].sum()), float(segs[1].sum())),
        tip_angle=math.degrees(math.atan2(z[0], z[2])),
        reversed=int(np.sum((ang * np.sign(sum(thetas))) < -0.5)),
        dl_mm=np.round((target - ctrl0) * 1000, 1),
        ctrl_margin_mm=float(margin.min() * 1000),
        peak_force_N=round(max_force[0], 2),
        force_limit_N=float(-m.actuator_forcerange[:, 0].max()),
        settled=qv < sw.SETTLE_TOL, t=d.time, m=m, d=d, ctrl0=ctrl0, target=target,
    )


def overdrive(xml, L, seg, sign):
    """Secant search on the commanded bend so the tip axis lands at sign*60."""
    def f(cmd):
        th = [0.0, 0.0]
        th[seg] = sign * math.radians(cmd)
        r = run(xml, L, th)
        return sign * r["tip_angle"] - TARGET, r

    a, b = TARGET, TARGET * 1.3
    fa, _ = f(a)
    fb, rb = f(b)
    for _ in range(8):
        if abs(fb) < 0.2 or fb == fa:
            break
        a, fa, b = b, fb, min(max(b - fb * (b - a) / (fb - fa), TARGET), 179.0)
        fb, rb = f(b)
    return b, rb


def main():
    global DISCS, GAP
    p = argparse.ArgumentParser()
    p.add_argument("--render", action="store_true", help="record the worse case at 60 deg command")
    p.add_argument("--discs", type=int, default=DISCS, help="discs per segment")
    p.add_argument("--gap", type=float, default=GAP, help="clear gap between disc faces (mm)")
    p.add_argument("--youngs-mpa", type=float, help="override tube Young's modulus (MPa) for this run only")
    args = p.parse_args()
    DISCS, GAP = args.discs, args.gap
    cfg, L = sd.make_config(DISCS, GAP + sw.DISC_T, LPP, args.youngs_mpa)
    E = json.load(open(cfg))["material_properties"]["youngs_modulus"] / 1e6
    xml = sw.generate(cfg, sw.HANG_EULER, sd.OUT)
    print(f"E = {E:g} MPa, 2 x {DISCS} discs, gap {GAP:g} mm, segment {L * 1000:g} mm, hanging, gravity on")

    results = []
    for seg in (0, 1):
        for sign, side in ((+1, "+x"), (-1, "-x")):
            th = [0.0, 0.0]
            th[seg] = sign * math.radians(TARGET)
            r = run(xml, L, th)
            od, ro = overdrive(xml, L, seg, sign)
            results.append((seg, side, sign, r))
            print(
                f"seg{seg + 1} {side}: cmd 60 -> seg bends {r['seg_bend'][0]:+.1f}/{r['seg_bend'][1]:+.1f} deg, "
                f"tip axis {r['tip_angle']:+.1f} deg (short {TARGET - sign * r['tip_angle']:.1f}), "
                f"reversed joints {r['reversed']}, settled {r['settled']} ({r['t']:.0f} s) | "
                f"overdrive cmd {od:.1f} deg -> tip {ro['tip_angle']:+.1f}, seg bends "
                f"{ro['seg_bend'][0]:+.1f}/{ro['seg_bend'][1]:+.1f}, max tendon pull "
                f"{-ro['dl_mm'].min():.1f} mm, ctrlrange margin {ro['ctrl_margin_mm']:.1f} mm, "
                f"peak force {ro['peak_force_N']} / {ro['force_limit_N']:g} N"
            )

    if args.render:
        seg, side, sign, r = max(results, key=lambda x: TARGET - x[2] * x[3]["tip_angle"])
        m, d = sd.build(xml, DISCS, LPP, True)
        th = [0.0, 0.0]
        th[seg] = sign * math.radians(TARGET)
        ctrl0 = d.ctrl.copy()
        target = cc_ctrl(m, d, ctrl0, L, th, DISCS)
        tag = "" if (DISCS, GAP) == (4, 30.0) else f"_d{DISCS}_gap{GAP:g}"
        tag += f"_E{args.youngs_mpa:g}MPa" if args.youngs_mpa is not None else ""
        out, n = sd.render(m, d, ctrl0, target, L,
                           sw.ROOT / f"outputs/bend60_seg{seg + 1}_{side.replace('+', 'pos').replace('-', 'neg')}{tag}")
        print(f"rendered worse case seg{seg + 1} {side}: {out} ({n} frames @ 25 fps)")


if __name__ == "__main__":
    main()
