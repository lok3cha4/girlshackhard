"""Headless render of a generated TDCR scene bending under tendon pulls.

Frames are sampled by simulated time (one frame every 1/fps sim-seconds),
not per physics step.

    uv run python scripts/render_bend.py assets/single_segment_4tendon_world_scene.xml outputs/single_segment_bend
"""

import argparse
import subprocess
from pathlib import Path

import mujoco
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scene")
    p.add_argument("out", help="output path without extension (.mp4 is written)")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--size", default="480x320")
    p.add_argument("--pull", type=float, default=0.01, help="tendon shortening (m)")
    p.add_argument("--seconds", type=float, default=6.0)
    args = p.parse_args()
    w, h = (int(v) for v in args.size.split("x"))

    m = mujoco.MjModel.from_xml_path(args.scene)
    d = mujoco.MjData(m)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "pretension")
    mujoco.mj_resetDataKeyframe(m, d, key)
    ctrl0 = d.ctrl.copy()

    m.vis.global_.offwidth = max(m.vis.global_.offwidth, w)
    m.vis.global_.offheight = max(m.vis.global_.offheight, h)
    renderer = mujoco.Renderer(m, height=h, width=w)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    mujoco.mj_forward(m, d)
    links = [i for i in range(m.nbody) if m.body(i).name.startswith("link_")]
    cam.lookat[:] = d.xpos[links].mean(axis=0)
    cam.distance, cam.azimuth, cam.elevation = 0.35, 120, -15

    # tendon 0 pulled during [0.5, 2.5) s, tendon 1 during [3.5, 5.5) s
    def target(t):
        u = d.ctrl.copy()
        u[:] = ctrl0
        for n, t0 in ((0, 0.5), (1, 3.5)):
            s = np.clip((t - t0) / 1.0, 0, 1) * np.clip((t0 + 2.0 - t) / 0.5, 0, 1)
            u[n] = ctrl0[n] - args.pull * s
        return u

    frames, next_frame, tips = [], 0.0, []
    tip = m.nbody - 1
    while d.time < args.seconds:
        d.ctrl[:] = target(d.time)
        mujoco.mj_step(m, d)
        if d.time >= next_frame:
            renderer.update_scene(d, cam)
            frames.append(renderer.render().copy())
            tips.append(d.xpos[tip].copy())
            next_frame += 1.0 / args.fps
        assert np.all(d.actuator_force <= 1e-9), "tendon pushed"

    out = Path(args.out).with_suffix(".mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(args.fps), "-i", "-", "-c:v", "libx264",
         "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p", str(out)],
        input=b"".join(f.tobytes() for f in frames), check=True,
    )
    tips = np.array(tips)
    print(f"wrote {out}: {len(frames)} frames @ {args.fps} fps, sim {d.time:.2f} s")
    print(f"tip range x[{tips[:,0].min():.3f},{tips[:,0].max():.3f}] "
          f"y[{tips[:,1].min():.3f},{tips[:,1].max():.3f}] z[{tips[:,2].min():.3f},{tips[:,2].max():.3f}]")
    renderer.close()


if __name__ == "__main__":
    main()
