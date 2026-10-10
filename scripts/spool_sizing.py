"""SG90 spool sizing for the (legacy) 2-segment disc design at a 60 deg demo bend.

The current hardware is 1 segment / 2 servos; use one_seg_check.py for it.

For each demo command (seg1 60, seg2 60, both 60, plus the overdrive needed to
land the tip axis at 60) this takes, from the hanging/gravity sim:
  * net load per servo = primary - antagonist tendon tension (peak during the
    ramp and steady state), and
  * tendon travel per servo (pair half-difference, serial_bridge kinematics),
then scales tension by a capstan friction estimate (not in the sim) and finds
the spool-radius band where travel fits the servo swing and the friction-
scaled peak stays under 50% of SG90 stall.

    uv run python scripts/spool_sizing.py --discs 4 --gap 10
"""

import argparse
import math

import numpy as np

import bend60_check as b
import serial_bridge as sb
import sim_disc_design as sd
import sweep_2seg as sw

STALL_NM = 1.2 * 9.80665 * 0.01  # SG90 stall, conservative 1.2 kgf cm
MU = 0.25  # tendon-on-PLA-hole friction, assumed (braided line ~0.15-0.3)
SAFE_FRAC = 0.5
# Legacy 2-segment analysis only (4 pairs over 2 segments). The real hardware
# is 1 segment / 2 servos: see serial_bridge.SERVO_MAP and one_seg_check.py.
PAIRS_2SEG = [(0, 0, 2), (0, 1, 3), (1, 0, 2), (1, 1, 3)]  # (segment, primary, antagonist)
TRAVEL_MARGIN = 1.10


def servo_loads(xml, L, discs, cmd_deg):
    th = [math.radians(t) for t in cmd_deg]
    b.DISCS = discs
    peak = np.zeros(4)
    m, d = sd.build(xml, discs, b.LPP, True)
    ctrl0 = d.ctrl.copy()
    target = b.cc_ctrl(m, d, ctrl0, L, th, discs)

    def net():
        F = -d.actuator_force
        return np.array([F[4 * s + p] - F[4 * s + a] for s, p, a in PAIRS_2SEG])

    def ctrl_fn(s):
        peak[:] = np.maximum(peak, np.abs(net()))
        return ctrl0 + s * (target - ctrl0)

    sw.settle(m, d, ctrl_fn, sw.SETTLE_MAX, tol=sw.SETTLE_TOL)
    return np.abs(net()), peak


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--discs", type=int, default=4)
    p.add_argument("--gap", type=float, default=10.0)
    p.add_argument("--youngs", type=float, nargs="+", default=[2.0, 3.0], help="MPa")
    p.add_argument("--overdrive", type=float, nargs=2, default=None,
                   metavar=("SEG1", "SEG2"), help="overdrive commands (deg) from bend60_check")
    p.add_argument("--safe-frac", type=float, nargs="+", default=[SAFE_FRAC],
                   help="allowed peak load as fraction of stall (e.g. 0.35 0.40)")
    p.add_argument("--swings", type=float, nargs="+", default=[30, 60, 80],
                   help="servo swing either side of 90 deg (e.g. 60 for [30,150])")
    p.add_argument("--one-at-a-time", action="store_true", help="skip the both-segments-at-60 case")
    args = p.parse_args()

    pitch = args.gap + sw.DISC_T
    cases = {"seg1 60": (60, 0), "seg2 60": (0, 60)}
    if not args.one_at_a_time:
        cases["both 60"] = (60, 60)
    if args.overdrive:
        cases[f"seg1 {args.overdrive[0]:g} (od)"] = (args.overdrive[0], 0)
        cases[f"seg2 {args.overdrive[1]:g} (od)"] = (0, args.overdrive[1])

    cfg, L = sd.make_config(args.discs, pitch, b.LPP)
    segs = sb.load_layout(cfg)
    travel = {}
    for name, (t1, t2) in cases.items():
        dl = sb.tendon_dl(segs, [(math.radians(t1), 0.0), (math.radians(t2), 0.0)], args.discs)
        travel[name] = np.array([abs(dl[(s, a)] - dl[(s, pr)]) / 2 for s, pr, a in PAIRS_2SEG]) * 1000
    print(f"layout: 2 x {args.discs} discs, gap {args.gap:g} mm, segment {L * 1000:g} mm; "
          f"SG90 stall {STALL_NM:.4f} N m; friction mu={MU} capstan (assumed)")

    worst = {}
    for E in args.youngs:
        cfgE, _ = sd.make_config(args.discs, pitch, b.LPP, E)
        xml = sw.generate(cfgE, sw.HANG_EULER, sd.OUT)
        for name, (t1, t2) in cases.items():
            steady, peak = servo_loads(xml, L, args.discs, (t1, t2))
            wrap = [math.radians(t1), math.radians(t1 + t2)]  # seg1 / seg2 tendons
            fric = np.array([math.exp(MU * wrap[s]) for s, _, _ in PAIRS_2SEG])
            pk_f = peak * fric
            worst[(E, name)] = (peak.max(), pk_f.max(), travel[name].max())
            print(f"E {E:g} MPa | {name:16s}: steady {np.round(steady, 2)} N, peak {np.round(peak, 2)} N, "
                  f"x friction {np.round(pk_f, 2)} N | travel {np.round(travel[name], 1)} mm")
            worst[(E, name, "peak_vec")] = (peak, pk_f)

    scalar = {k: v for k, v in worst.items() if len(k) == 2}
    pk_worst = max(v[1] for v in scalar.values())
    scopes = [("one segment at a time", [n for n in cases if "both" not in n])]
    if not args.one_at_a_time:
        scopes.append(("incl. both at 60", list(cases)))
    for frac in args.safe_frac:
        print(f"\nspool radius band (travel x{TRAVEL_MARGIN} must fit swing; friction-scaled peak <= {frac:.0%} stall):")
        for scope, names in scopes:
            need = max(travel[n].max() for n in names) * TRAVEL_MARGIN
            pk = max(scalar[(E, n)][1] for E in args.youngs for n in names)
            r_max = STALL_NM * frac / pk * 1000
            for swing in args.swings:
                r_min = need / math.radians(swing)
                ok = "OK" if r_min <= r_max else "NO FIT"
                load_now = pk / (STALL_NM / sb.R_SPOOL)
                print(f"  {scope:22s} swing +/-{swing:g} deg: r_min {r_min:5.1f} mm (travel {need:.1f} mm), "
                      f"r_max {r_max:5.1f} mm (peak {pk:.2f} N w/ friction) -> {ok}; "
                      f"at current r_spool {sb.R_SPOOL * 1000:g} mm load = {load_now:.0%} of stall")
    print(f"worst friction-scaled peak over all cases/E: {pk_worst:.2f} N")


if __name__ == "__main__":
    main()
