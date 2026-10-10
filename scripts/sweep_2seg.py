"""Headless 2-segment sweep over discs (links_per_segment) x segment length.

Derives each config from configs/generation/single_segment_4tendon.json
(silicone tube material, top/hanging mount), forces 2 equal segments, low
pretension (0.1 N) / kp 1000, generates it with third_party generate.py into
outputs/sweep_2seg/, and measures mass, gravity sag, 45 deg + 45 deg
constant-curvature tip error, and S-buckling. Writes outputs/sweep_2seg.csv.

    uv run python scripts/sweep_2seg.py
    uv run python scripts/sweep_2seg.py --render 8 0.12   # one video
    uv run python scripts/sweep_2seg.py --configs 8:0.08 --with-discs --disc-diameter 15
"""

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GEN_DIR = ROOT / "third_party/opencr-mujoco"
BASE_CFG = ROOT / "configs/generation/single_segment_4tendon.json"
OUT = ROOT / "outputs/sweep_2seg"

DISCS = [5, 8, 12, 16]
LENGTHS = [0.08, 0.12, 0.16]
THETA = math.radians(45.0)
HANG_EULER, SIDE_EULER = "0 3.1416 0", "0 1.5708 0"

# PLA disc (user geometry): 40 mm OD (default, --disc-diameter), 3 mm thick,
# 8.5 mm backbone hole, 1.75 mm tendon holes whose edge sits 2.125 mm in from
# the disc rim; 15% grid infill. Slicer assumptions are rough.
PLA_DENSITY = 1.24  # g/cm^3
DISC_OD, DISC_T, BB_HOLE, TEN_HOLE = 40.0, 3.0, 8.5, 1.75  # mm
RIM_TO_TENDON_EDGE = 2.125  # mm
# mm, distance_to_backbone written into every sweep config (tendon-hole centre)
TENDON_OFFSET = DISC_OD / 2 - RIM_TO_TENDON_EDGE - TEN_HOLE / 2
INFILL, WALL, SKIN = 0.15, 0.9, 0.8  # 2 x 0.45 mm perimeters, 4 x 0.2 mm top/bottom
FLOW = 3.0  # mm^3/s average effective extrusion rate incl. travel (stock 0.4 mm nozzle)


def disc_estimate(n_holes, od=DISC_OD, bb=BB_HOLE):
    area = math.pi / 4 * (od**2 - bb**2 - n_holes * TEN_HOLE**2)
    perim = math.pi * (od + bb + n_holes * TEN_HOLE)
    wall_area = min(area, perim * WALL)
    core = wall_area + INFILL * (area - wall_area)
    vol = area * 2 * SKIN + core * (DISC_T - 2 * SKIN)  # mm^3 of plastic
    return vol * PLA_DENSITY / 1000, vol / FLOW / 60  # g, min


def disc_walls(od=DISC_OD, bb=BB_HOLE, offset=TENDON_OFFSET):
    """Radial wall thicknesses (mm) of a disc; flags any below one perimeter pair.

    total: disc radius - backbone-hole radius - tendon-hole radius (all the
    radial material, ignoring where the tendon hole sits).
    inner/outer: the actual webs either side of a tendon hole at `offset`.
    """
    r_th = TEN_HOLE / 2
    w = dict(wall_total_mm=od / 2 - bb / 2 - r_th,
             wall_inner_mm=offset - bb / 2 - r_th,
             wall_outer_mm=od / 2 - offset - r_th)
    w = {k: round(v, 2) for k, v in w.items()}
    w["wall_too_thin"] = ",".join(k[5:-3] for k, v in w.items() if v < WALL) or "no"
    return w


def make_config(n, L):
    c = json.load(open(BASE_CFG))
    name = f"sweep2_n{n}_L{int(round(L * 1000))}"
    c.update(
        description=f"2-segment sweep point: {n} discs/segment, {L} m/segment",
        num_segments=2,
        links_per_segment={"1": n, "2": n},
        segment_lengths={"1": L, "2": L},
        seg_offsets=[0, math.pi / 4],
    )
    c["actuation_details"] = {
        "segments": [{"number_of_tendons": 4, "distance_to_backbone": TENDON_OFFSET / 1000}] * 2
    }
    c["actuator_properties"].update(tendon_pretension=0.1, tendon_kp=1000)
    c.pop("tendon_angle_deltas", None)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.json"
    json.dump(c, open(path, "w"), indent=2)
    return name, path


def generate(cfg_path, euler, outdir):
    subprocess.run(
        ["uv", "run", "python", "generate.py", "--config-file", str(cfg_path),
         "--world", "--mount-euler", euler, "--output-dir", str(outdir)],
        cwd=GEN_DIR, check=True, capture_output=True,
    )
    return outdir / f"{cfg_path.stem}_world_scene.xml"


def add_discs(spec, n, od=DISC_OD, bb=BB_HOLE):
    """Weld a PLA disc (own mass + annulus inertia) at each moving link's centre.

    The generator gives 2n+1 link bodies (half links at both ends); link_0 is
    rigidly fixed to the mount, so links 1..2n carry the 2n physical discs.
    Links 1..n get segment-1 discs (8 tendon holes), n+1..2n segment-2 (4).
    The tube's own mass from the generator is left untouched.
    """
    ro, ri, h = od / 2000, bb / 2000, DISC_T / 1000
    for k in range(1, 2 * n + 1):
        link = spec.body(f"link_{k}")
        mass = disc_estimate(8 if k <= n else 4, od, bb)[0] / 1000
        i_ax = mass / 2 * (ro**2 + ri**2)
        i_tr = mass / 12 * (3 * (ro**2 + ri**2) + h**2)
        ft = np.asarray(link.first_geom().fromto)  # generator writes link geoms as fromto
        centre = (ft[:3] + ft[3:]) / 2
        disc = link.add_body(name=f"disc_{k}", pos=centre, explicitinertial=True,
                             mass=mass, inertia=[i_tr, i_tr, i_ax])
        disc.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[ro, h / 2, 0],
                      contype=0, conaffinity=0, rgba=[0.95, 0.6, 0.2, 1])


def load(xml, discs=0, od=DISC_OD, bb=BB_HOLE):
    spec = mujoco.MjSpec.from_file(str(xml))
    if discs:
        add_discs(spec, discs, od, bb)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("pretension").id)
    mujoco.mj_forward(m, d)
    return m, d


def mount_frame(m, d):
    i = m.body("tdcr_mount").id
    return d.xpos[i].copy(), d.xmat[i].reshape(3, 3).copy()


def to_mount(m, d, p):
    o, R = mount_frame(m, d)
    return R.T @ (p - o)


def cc_tip(L, thetas):
    """Composed constant-curvature tip, in the robot base (mount) frame."""
    p, R = np.zeros(3), np.eye(3)
    for th in thetas:
        r = L / th
        p = p + R @ np.array([r * (1 - math.cos(th)), 0.0, r * math.sin(th)])
        R = R @ np.array([[math.cos(th), 0, math.sin(th)], [0, 1, 0],
                          [-math.sin(th), 0, math.cos(th)]])
    return p


def cc_ctrl(m, d, ctrl0, thetas):
    """Tendon length commands for an in-plane (mount x-z) CC bend.

    dl_i = -(sum of theta over the segments the tendon spans) * (r_i . x_mount),
    with r_i the tendon's offset from the backbone at rest (coupled routing:
    segment-s tendons run through every segment <= s).
    """
    u = ctrl0.copy()
    for a in range(m.nu):
        t = m.actuator_trnid[a, 0]
        s = int(m.tendon(t).name.split("_")[1])
        site = m.wrap_objid[m.tendon_adr[t]]
        x_off = to_mount(m, d, d.site_xpos[site])[0]
        u[a] = ctrl0[a] - sum(thetas[: s + 1]) * x_off
    return u


def bend_joints(m, d):
    """Per-link in-plane bend angles (deg), signed so +x bending is positive."""
    _, R = mount_frame(m, d)
    y = R[:, 1]
    out = []
    for j in range(m.njnt):
        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        ax = d.xaxis[j]
        if abs(ax @ y) > 0.9:
            out.append((j, math.copysign(1, ax @ y)))
    return out


def settle(m, d, ctrl_fn, seconds, ramp=1.0, tol=None):
    """Step until `seconds`, or (with tol) until |qvel| < tol after the ramp."""
    while d.time < seconds:
        d.ctrl[:] = ctrl_fn(min(1.0, d.time / ramp))
        mujoco.mj_step(m, d)
        assert np.all(d.actuator_force <= 1e-9), "tendon pushed"
        if tol and d.time > ramp + 1 and np.abs(d.qvel).max() < tol:
            break
    return float(np.abs(d.qvel).max())


SETTLE_TOL, SETTLE_MAX = 5e-4, 120.0


def gap_check(n, L, od=DISC_OD):
    """Disc gap (mm) and the per-segment bend (deg) at which disc rims touch."""
    gap = L * 1000 / n - DISC_T
    return gap, math.degrees(n * gap / (od / 2)) if gap > 0 else 0.0


def mass_cols(m, row):
    names = [m.body(i).name for i in range(m.nbody)]
    pick = lambda pre: sum(m.body_mass[i] for i, nm in enumerate(names) if nm.startswith(pre))
    row["tube_mass_g"] = round(pick("link_") * 1000, 2)
    row["disc_mass_g"] = round(pick("disc_") * 1000, 2)
    row["total_mass_g"] = round(row["tube_mass_g"] + row["disc_mass_g"], 2)


def measure(n, L, discs=False, od=DISC_OD, bb=BB_HOLE):
    name, cfg = make_config(n, L)
    nd = n if discs else 0
    row = dict(discs_per_seg=n, seg_len_m=L, total_len_m=2 * L, total_discs=2 * n,
               mass_model="tube+PLA discs" if discs else "tube only (lumped)")
    row.update(disc_od_mm=od, bb_hole_mm=bb, **disc_walls(od, bb))
    if row["wall_too_thin"] != "no":
        print(f"WARNING: {od} mm disc wall(s) below {WALL} mm: {row['wall_too_thin']}")
    row["gap_mm"], row["rim_contact_bend_deg"] = (round(v, 1) for v in gap_check(n, L, od))

    # side mount: gravity sag at zero command (same metric as single-segment)
    m, d = load(generate(cfg, SIDE_EULER, OUT / "side"), nd, od, bb)
    tip0 = d.xpos[m.body("EE_pos").id].copy()
    settle(m, d, lambda s, c=d.ctrl.copy(): c, SETTLE_MAX, tol=SETTLE_TOL)
    l0, l1 = (d.xpos[m.body(b).id] for b in ("link_1", "link_2"))
    tan = to_mount(m, d, l1) - to_mount(m, d, l0)
    row["side_base_angle_deg"] = round(math.degrees(math.atan2(math.hypot(tan[0], tan[1]), tan[2])), 1)
    row["side_tip_drop_mm"] = round((tip0[2] - d.xpos[m.body("EE_pos").id][2]) * 1000, 1)

    # top (hanging) mount: mass, 45+45 deg CC bend, with and without gravity
    xml = generate(cfg, HANG_EULER, OUT)
    for grav in (True, False):
        m, d = load(xml, nd, od, bb)
        if not grav:
            m.opt.gravity[:] = 0
        mass_cols(m, row)
        ctrl0 = d.ctrl.copy()
        target = cc_ctrl(m, d, ctrl0, [THETA, THETA])
        qv = settle(m, d, lambda s: ctrl0 + s * (target - ctrl0), SETTLE_MAX, tol=SETTLE_TOL)
        tip = to_mount(m, d, d.xpos[m.body("EE_pos").id])
        err = np.linalg.norm(tip - cc_tip(L, [THETA, THETA])) * 1000
        key = "g" if grav else "0g"
        row[f"tip_err_{key}_mm"] = round(err, 1)
        row[f"tip_err_{key}_pct_len"] = round(err / (2 * L * 1000) * 100, 1)
        bj = bend_joints(m, d)
        ang = np.array([math.degrees(d.qpos[m.jnt_qposadr[j]]) * s for j, s in bj])
        segs = np.array_split(ang, 2)
        seg_bend = [round(float(a.sum()), 1) for a in segs]
        row[f"seg_bend_{key}_deg"] = f"{seg_bend[0]}/{seg_bend[1]}"
        neg = int(np.sum(ang < -0.5))
        cv = max(float(np.std(a) / max(abs(np.mean(a)), 1e-6)) for a in segs)
        row[f"reversed_joints_{key}"] = neg
        row[f"curv_cv_{key}"] = round(cv, 2)
        row[f"shape_{key}"] = "S" if neg else ("arc" if cv < 0.25 else "non-uniform arc")
        row[f"settle_t_{key}_s"] = round(d.time, 1)
        row[f"settled_{key}"] = qv < SETTLE_TOL

    for holes, tag in ((8, "seg1"), (4, "seg2")):
        g, mins = disc_estimate(holes, od, bb)
        row[f"pla_g_per_disc_{tag}"] = round(g, 2)
        row.setdefault("_pla", []).append((g, mins))
    (g1, t1), (g2, t2) = row.pop("_pla")
    row["pla_total_g"] = round(n * (g1 + g2), 1)
    row["print_time_min"] = round(n * (t1 + t2))
    return row, xml


def render(xml, L, out, fps=25, size=(480, 320), seconds=6.0, discs=0, od=DISC_OD):
    m, d = load(xml, discs, od)
    ctrl0 = d.ctrl.copy()
    target = cc_ctrl(m, d, ctrl0, [THETA, THETA])
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, size[0])
    m.vis.global_.offheight = max(m.vis.global_.offheight, size[1])
    r = mujoco.Renderer(m, height=size[1], width=size[0])
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    links = [i for i in range(m.nbody) if m.body(i).name.startswith("link_")]
    cam.lookat[:] = d.xpos[links].mean(axis=0)
    cam.distance, cam.azimuth, cam.elevation = 0.6 * 2 * L / 0.24, 90, -5
    frames, nxt = [], 0.0
    while d.time < seconds:
        s = np.clip((d.time - 0.5) / 1.5, 0, 1) * np.clip((5.5 - d.time) / 1.0, 0, 1)
        d.ctrl[:] = ctrl0 + s * (target - ctrl0)
        mujoco.mj_step(m, d)
        if d.time >= nxt:
            r.update_scene(d, cam)
            frames.append(r.render().copy())
            nxt += 1.0 / fps
    out = Path(out).with_suffix(".mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
         "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p", str(out)],
        input=b"".join(f.tobytes() for f in frames), check=True,
    )
    r.close()
    print(f"wrote {out}: {len(frames)} frames @ {fps} fps")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--render", nargs=2, metavar=("DISCS", "LEN"))
    p.add_argument("--configs", nargs="+", metavar="DISCS:LEN", help="subset, e.g. 8:0.08 16:0.08")
    p.add_argument("--with-discs", action="store_true", help="add PLA disc mass/inertia via MjSpec")
    p.add_argument("--csv", default="outputs/sweep_2seg.csv")
    p.add_argument("--disc-diameter", type=float, default=DISC_OD,
                   help=f"PLA disc outer diameter in mm (default {DISC_OD:g})")
    args = p.parse_args()
    if args.render:
        n, L = int(args.render[0]), float(args.render[1])
        name, cfg = make_config(n, L)
        tag = "_discs" if args.with_discs else ""
        render(generate(cfg, HANG_EULER, OUT), L, ROOT / f"outputs/{name}{tag}_bend",
               discs=n if args.with_discs else 0, od=args.disc_diameter)
        return
    grid = ([(int(a), float(b)) for a, b in (c.split(":") for c in args.configs)]
            if args.configs else [(n, L) for L in LENGTHS for n in DISCS])
    rows = []
    for n, L in grid:
        for discs in ((False, True) if args.with_discs else (False,)):
            row, _ = measure(n, L, discs, od=args.disc_diameter)
            rows.append(row)
            print(row)
    with open(ROOT / args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
