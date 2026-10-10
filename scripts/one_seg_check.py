"""1-segment bend + SG90 load check for configs/generation/one_segment_8disc.json.

Hanging mount, gravity, real PLA disc mass (MjSpec, 4-hole discs), tendons
routed through the disc holes only. Tendon commands come from serial_bridge's
own kinematics (what the pitch/yaw UI sends), so the sim sees the same
lengths the servos would produce. For each (bend, direction) it reports the
achieved tip-axis bend, an arc-vs-S shape check, settling time, and per-servo
load (primary - antagonist tension) vs SG90 stall at R_SPOOL, with and
without the capstan friction estimate. Overdrive search lands the tip at the
commanded angle.

    uv run python scripts/one_seg_check.py
"""

import argparse
import json
import math

import mujoco
import numpy as np

import serial_bridge as sb
import sim_disc_design as sd
import sweep_2seg as sw

CONFIG = sb.DEFAULT_CONFIG
OUT = sb.ROOT / "outputs/one_seg"
DISCS, LPP = sb.DEFAULT_DISCS, 3
STALL_NM = 1.2 * 9.80665 * 0.01  # SG90, conservative 1.2 kgf cm
MU = 0.25  # assumed tendon/PLA-hole friction (not in the sim)


def make_xml(youngs_mpa):
    c = json.load(open(CONFIG))
    c["material_properties"]["youngs_modulus"] = youngs_mpa * 1e6
    assert c["links_per_segment"]["1"] == DISCS * LPP, "links must be DISCS x LPP"
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = OUT / f"one_segment_8disc_E{youngs_mpa:g}MPa.json"
    json.dump(c, open(cfg, "w"), indent=2)
    return sw.generate(cfg, sw.HANG_EULER, OUT), c["segment_lengths"]["1"]


def run(xml, theta_deg, phi_deg):
    m, d = sd.build(xml, DISCS, LPP, True, n_segments=1)
    segs = sb.load_layout(CONFIG)
    dl = sb.tendon_dl(segs, [(math.radians(theta_deg), math.radians(phi_deg))], DISCS)
    ctrl0 = d.ctrl.copy()
    target = ctrl0 + np.array([dl[(0, int(m.actuator(a).name.split("_")[3]))] for a in range(m.nu)])
    peak = np.zeros(m.nu)

    def ctrl_fn(s):
        peak[:] = np.maximum(peak, -d.actuator_force)
        return ctrl0 + s * (target - ctrl0)

    qv = sw.settle(m, d, ctrl_fn, sw.SETTLE_MAX, tol=sw.SETTLE_TOL)
    o, R = sw.mount_frame(m, d)
    z = R.T @ d.xmat[m.body("EE_pos").id].reshape(3, 3)[:, 2]
    tip_theta = math.degrees(math.acos(max(-1.0, min(1.0, z[2]))))
    tip_phi = math.degrees(math.atan2(z[1], z[0]))

    # in-plane bend per link: joint angles projected on the bend axis
    phi = math.radians(phi_deg)
    b = R @ np.array([-math.sin(phi), math.cos(phi), 0.0])
    per_link = {}
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            body = m.jnt_bodyid[j]
            per_link[body] = per_link.get(body, 0.0) + d.qpos[m.jnt_qposadr[j]] * float(d.xaxis[j] @ b)
    ang = np.degrees(np.array(list(per_link.values())))
    rev = int(np.sum(ang < -0.5))
    cv = float(np.std(ang) / max(abs(np.mean(ang)), 1e-6))

    F = -d.actuator_force
    idx = {int(m.actuator(a).name.split("_")[3]): a for a in range(m.nu)}
    net = lambda v: [abs(v[idx[0]] - v[idx[2]]), abs(v[idx[1]] - v[idx[3]])]
    fric = math.exp(MU * math.radians(theta_deg))
    servo_deg, raw, _ = sb.servo_angles(segs, math.radians(theta_deg), math.radians(phi_deg), DISCS)
    return dict(
        tip_theta=tip_theta, tip_phi=tip_phi, inplane=float(ang.sum()), reversed=rev, cv=cv,
        shape="S" if rev else ("arc" if cv < 0.25 else "non-uniform arc"),
        settled=qv < sw.SETTLE_TOL, t=d.time,
        steady=net(F), peak=net(peak), peak_f=[v * fric for v in net(peak)],
        servo=[round(v) for v in servo_deg],
        clamped=any(abs(r - c) > 1e-9 for r, c in zip(raw, servo_deg)),
    )


def overdrive(xml, theta, phi):
    """Secant search on the command so the tip axis lands at `theta`."""
    a, b = theta, theta * 1.1
    fa = run(xml, a, phi)["tip_theta"] - theta
    rb = run(xml, b, phi)
    fb = rb["tip_theta"] - theta
    for _ in range(8):
        if abs(fb) < 0.2 or fb == fa:
            break
        a, fa, b = b, fb, max(theta, b - fb * (b - a) / (fb - fa))
        rb = run(xml, b, phi)
        fb = rb["tip_theta"] - theta
    return b, rb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--youngs", type=float, nargs="+", default=[2.0, 3.0], help="MPa")
    args = p.parse_args()
    stall_T = STALL_NM / sb.R_SPOOL
    L = json.load(open(CONFIG))["segment_lengths"]["1"]
    print(f"1 segment x {DISCS} discs, {L * 1000:g} mm, pitch {L / DISCS * 1000:g} mm "
          f"(gap {L / DISCS * 1000 - sw.DISC_T:g} mm), hanging, gravity; "
          f"SG90 stall tension at r_spool {sb.R_SPOOL * 1000:g} mm = {stall_T:.2f} N; friction mu={MU} (assumed)")
    cases = [("pitch 60 (slider max)", 60.0, 0.0), ("yaw -60", 60.0, -90.0),
             ("diag 60/60 (UI corner)", math.hypot(60, 60), 45.0)]
    for E in args.youngs:
        xml, _ = make_xml(E)
        for name, th, ph in cases:
            r = run(xml, th, ph)
            od, ro = overdrive(xml, th, ph)
            pk = max(r["peak_f"] + ro["peak_f"])
            print(
                f"E {E:g} | {name:22s}: tip {r['tip_theta']:5.1f} deg @ {r['tip_phi']:+6.1f} "
                f"(short {th - r['tip_theta']:.1f}), shape {r['shape']} (reversed {r['reversed']}, cv {r['cv']:.2f}), "
                f"settled {r['settled']} {r['t']:.1f} s | servos {r['servo']}{' CLAMPED' if r['clamped'] else ''} | "
                f"load steady {np.round(r['steady'], 2)} N, peak {np.round(r['peak'], 2)} N, "
                f"x friction {np.round(r['peak_f'], 2)} N | overdrive {od:.1f} -> tip {ro['tip_theta']:.1f}, "
                f"servos {ro['servo']}{' CLAMPED' if ro['clamped'] else ''}, worst load {pk:.2f} N = {pk / stall_T:.0%} stall"
            )


if __name__ == "__main__":
    main()
