"""Simulate one 2-segment disc design: discs at a fixed pitch on a finely
discretised silicone tube, tendons routed only through the disc holes.

The generator (third_party, untouched) builds the tube with `links_per_pitch`
rigid links between consecutive discs. An MjSpec wrapper then
  * welds a PLA disc (mass + annulus inertia) at each disc position, and
  * re-routes every tendon through the disc-hole sites only (base, then every
    disc it passes), so tendons run in straight chords between discs as on
    the real robot instead of hugging the tube at every link.

Measures mass, side-mount gravity sag, and the 45 deg + 45 deg composed
constant-curvature tip error (with / without gravity), then records one
hanging 25 fps video.

    uv run python scripts/sim_disc_design.py --discs 4 --gap 30
"""

import argparse
import json
import math
import subprocess
from pathlib import Path

import mujoco
import numpy as np

import sweep_2seg as sw

ROOT = sw.ROOT
OUT = ROOT / "outputs/disc_design"


def make_config(discs, pitch_mm, links_per_pitch, youngs_mpa=None):
    c = json.load(open(sw.BASE_CFG))
    L = discs * pitch_mm / 1000
    n_links = discs * links_per_pitch
    name = f"design_d{discs}_p{pitch_mm:g}_lpp{links_per_pitch}"
    if youngs_mpa is not None:
        c["material_properties"]["youngs_modulus"] = youngs_mpa * 1e6
        name += f"_E{youngs_mpa:g}MPa"
    c.update(
        description=f"2 segments x {discs} discs at {pitch_mm:g} mm pitch "
        f"({L * 1000:g} mm/segment), {n_links} tube links/segment",
        num_segments=2,
        links_per_segment={"1": n_links, "2": n_links},
        segment_lengths={"1": L, "2": L},
        seg_offsets=[0, math.pi / 4],
    )
    c["actuation_details"] = {
        "segments": [{"number_of_tendons": 4, "distance_to_backbone": sw.TENDON_OFFSET / 1000}] * 2
    }
    c["actuator_properties"].update(tendon_pretension=0.1, tendon_kp=1000)
    c.pop("tendon_angle_deltas", None)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.json"
    json.dump(c, open(path, "w"), indent=2)
    return path, L


def build(xml, discs, links_per_pitch, with_discs=True, n_segments=2):
    """Compile the generated scene with discs added and tendons re-routed."""
    spec = mujoco.MjSpec.from_file(str(xml))
    ro, ri, h = sw.DISC_OD / 2000, sw.BB_HOLE / 2000, sw.DISC_T / 1000
    disc_links = [j * links_per_pitch for j in range(1, n_segments * discs + 1)]
    for j, k in enumerate(disc_links, start=1):
        link = spec.body(f"link_{k}")
        ft = np.asarray(link.first_geom().fromto)
        # disc sits at arc = k * link_length: the far end of a half-length tip
        # link, the midpoint of every full link (same place as the tendon site)
        pos = ft[3:] if k == disc_links[-1] else (ft[:3] + ft[3:]) / 2
        if with_discs:
            # proximal discs also carry the distal segment's tendons (8 holes)
            mass = sw.disc_estimate(8 if j <= discs * (n_segments - 1) else 4)[0] / 1000
            i_ax = mass / 2 * (ro**2 + ri**2)
            i_tr = mass / 12 * (3 * (ro**2 + ri**2) + h**2)
            disc = link.add_body(name=f"disc_{j}", pos=pos, explicitinertial=True,
                                 mass=mass, inertia=[i_tr, i_tr, i_ax])
            disc.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[ro, h / 2, 0],
                          contype=0, conaffinity=0, rgba=[0.95, 0.6, 0.2, 0.9])

    for t in list(spec.tendons):
        seg, n = int(t.name.split("_")[1]), int(t.name.split("_")[3])
        keep = [f"l_0_base_seg_{seg}_tendon_{n}"] + [
            f"l_{k}_seg_{seg}_tendon_{n}" for k in disc_links[: (seg + 1) * discs]
        ]
        attrs = dict(name=t.name, width=t.width, rgba=t.rgba.copy(),
                     frictionloss=t.frictionloss, damping=t.damping, stiffness=t.stiffness)
        spec.delete(t)
        new = spec.add_tendon(**attrs)
        for site in keep:
            new.wrap_site(site)

    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("pretension").id)
    mujoco.mj_forward(m, d)
    return m, d


def cc_ctrl(m, d, ctrl0, L, discs, thetas):
    """Exact chord-length tendon commands for an in-plane (mount x-z) CC bend.

    Per segment a tendon at offset x (toward the bend) crosses `discs` chords,
    each subtending alpha = theta/discs on an arc of radius R = L/theta:
    length = discs * 2 (R - x) sin(alpha/2), versus L when straight.
    """
    u = ctrl0.copy()
    for a in range(m.nu):
        t = m.actuator_trnid[a, 0]
        seg = int(m.tendon(t).name.split("_")[1])
        x = sw.to_mount(m, d, d.site_xpos[m.wrap_objid[m.tendon_adr[t]]])[0]
        dl = 0.0
        for th in thetas[: seg + 1]:
            R = L / th
            dl += discs * 2 * (R - x) * math.sin(th / discs / 2) - L
        u[a] = ctrl0[a] + dl
    return u


def bend_metrics(m, d, L, thetas):
    tip = sw.to_mount(m, d, d.xpos[m.body("EE_pos").id])
    err = np.linalg.norm(tip - sw.cc_tip(L, thetas)) * 1000
    ang = np.array([math.degrees(d.qpos[m.jnt_qposadr[j]]) * s for j, s in sw.bend_joints(m, d)])
    segs = np.array_split(ang, 2)
    cv = max(float(np.std(a) / max(abs(np.mean(a)), 1e-6)) for a in segs)
    neg = int(np.sum(ang < -0.5))
    return dict(tip_err_mm=round(err, 1), tip_err_pct=round(err / (2 * L * 1000) * 100, 1),
                seg_bend_deg=f"{segs[0].sum():.1f}/{segs[1].sum():.1f}",
                reversed_joints=neg, curv_cv=round(cv, 2),
                shape="S" if neg else ("arc" if cv < 0.25 else "non-uniform arc"))


def render(m, d, ctrl0, target, L, out, fps=25, size=(480, 320), seconds=10.0):
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, size[0])
    m.vis.global_.offheight = max(m.vis.global_.offheight, size[1])
    r = mujoco.Renderer(m, height=size[1], width=size[0])
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    o, _ = sw.mount_frame(m, d)
    cam.lookat[:] = o + np.array([0.06, 0, -L * 0.8])
    cam.distance, cam.azimuth, cam.elevation = 4.2 * L, 90, -5
    frames, nxt = [], 0.0
    while d.time < seconds:
        s = np.clip((d.time - 0.5) / 2.0, 0, 1) * np.clip((9.5 - d.time) / 1.5, 0, 1)
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
    return out, len(frames)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--discs", type=int, default=4, help="discs per segment")
    p.add_argument("--gap", type=float, default=30.0, help="clear gap between disc faces (mm)")
    p.add_argument("--links-per-pitch", type=int, default=3, help="tube links between discs")
    p.add_argument("--bend-deg", type=float, default=45.0, help="CC bend per segment")
    p.add_argument("--youngs-mpa", type=float, help="override tube Young's modulus (MPa) for this run only")
    args = p.parse_args()

    pitch = args.gap + sw.DISC_T
    cfg, L = make_config(args.discs, pitch, args.links_per_pitch, args.youngs_mpa)
    thetas = [math.radians(args.bend_deg)] * 2
    print(f"design: 2 x {args.discs} discs, pitch {pitch:g} mm (gap {args.gap:g}), "
          f"segment {L * 1000:g} mm, total {2 * L * 1000:g} mm, "
          f"tube link {pitch / args.links_per_pitch:g} mm, E "
          f"{json.load(open(cfg))['material_properties']['youngs_modulus'] / 1e6:g} MPa")
    print("disc walls:", sw.disc_walls(), f"| rims touch at "
          f"{math.degrees(args.discs * args.gap / (sw.DISC_OD / 2)):.0f} deg/segment")

    hang = sw.generate(cfg, sw.HANG_EULER, OUT)
    side = sw.generate(cfg, sw.SIDE_EULER, OUT / "side")

    for with_discs in (False, True):
        label = "tube + PLA discs" if with_discs else "tube only"
        m, d = build(side, args.discs, args.links_per_pitch, with_discs)
        row = {}
        sw.mass_cols(m, row)
        tip0 = d.xpos[m.body("EE_pos").id].copy()
        sw.settle(m, d, lambda s, c=d.ctrl.copy(): c, sw.SETTLE_MAX, tol=sw.SETTLE_TOL)
        row["side_tip_drop_mm"] = round((tip0[2] - d.xpos[m.body("EE_pos").id][2]) * 1000, 1)
        for grav in (True, False):
            m, d = build(hang, args.discs, args.links_per_pitch, with_discs)
            if not grav:
                m.opt.gravity[:] = 0
            ctrl0 = d.ctrl.copy()
            target = cc_ctrl(m, d, ctrl0, L, args.discs, thetas)
            qv = sw.settle(m, d, lambda s: ctrl0 + s * (target - ctrl0), sw.SETTLE_MAX, tol=sw.SETTLE_TOL)
            met = bend_metrics(m, d, L, thetas)
            met["settled"] = f"{qv < sw.SETTLE_TOL} ({d.time:.0f} s)"
            row["gravity" if grav else "zero_g"] = met
        print(f"[{label}]", row)

    m, d = build(hang, args.discs, args.links_per_pitch, True)
    ctrl0 = d.ctrl.copy()
    target = cc_ctrl(m, d, ctrl0, L, args.discs, thetas)
    print("tendon commands (mm):", np.round((target - ctrl0) * 1000, 1))
    out, n = render(m, d, ctrl0, target, L, ROOT / f"outputs/{cfg.stem}_bend")
    print(f"wrote {out}: {n} frames @ 25 fps")


if __name__ == "__main__":
    main()
