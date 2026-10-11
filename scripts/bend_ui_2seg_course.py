"""Anatomy-like obstacle course for the 2-segment robot (builds on bend_ui_2seg_z.py).

A winding, vessel-like tube (entry funnel -> bend -> narrowing -> bend back ->
target) hangs below the robot. You insert the robot with the Z slider (down =
advance) and steer both segments with the pitch/yaw sliders so it gets through
without jamming. The tube walls are real collision geometry; the 40 mm PLA
discs collide with them (as well as the tube links), so clearances are real.

The tube is scaled up so the 40 mm discs fit: inner diameter 90 mm, narrowing
to 64 mm. MuJoCo can't collide with a hollow (concave) mesh, so each wall
section is a ring of thin boxes (staves) along a smoothed centreline.

Collision groups: tube links contype 1 (self-collision already excluded by the
generator), discs contype 2 (they don't touch each other or the tube), walls
conaffinity 1|2 (touch both). The floor is moved out of the way.

Feedback: wall-contact count and total contact force (orange when touching),
progress along the course, tip distance to the green target, and a timer.
The camera follows the robot.

    uv run python scripts/bend_ui_2seg_course.py
Keys: as bend_ui_2seg_z.py (arrows / WASD steer, R/F up/down, [ ] stiffness,
space centers bends, H homes Z, G gravity), plus T = reset timer.
"""

import argparse
import math
import time

import mujoco
import numpy as np

import bend_ui as ui
import bend_ui_2seg as u2
import bend_ui_2seg_z as uz

WALL_T = 0.004          # wall (stave) thickness, m
N_STAVES = 24           # boxes around the circumference
SEG_LEN = 0.02          # centreline sample spacing, m
DISC_GROUP, LINK_GROUP = 2, 1
WALL_RGBA = [0.92, 0.55, 0.55, 0.28]

# Centreline relative to the straight robot's starting tip (x, z) in metres and
# inner radius at each control point: entry funnel, bend right, narrowing,
# bend back, exit to the target. Total depth 0.28 m, so the mount (which stays
# outside) never has to enter the tube with a 0.32 m robot.
COURSE = [
    # x,      z,     radius
    (0.000,  0.040, 0.065),
    (0.000, -0.020, 0.045),
    (0.000, -0.060, 0.045),
    (0.030, -0.120, 0.045),
    (0.055, -0.170, 0.034),   # narrowing
    (0.040, -0.215, 0.034),
    (0.010, -0.255, 0.045),
    (0.000, -0.290, 0.045),
]


def _catmull_rom(pts, step):
    """Densify control points (N x 3) along a Catmull-Rom spline, ~step apart."""
    pts = np.asarray(pts, float)
    ext = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])
    out = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1:i + 3]
        n = max(2, int(np.ceil(np.linalg.norm(p2[:2] - p1[:2]) / step)))
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(pts[-1])
    return np.array(out)


def course_centreline(tip0):
    """World centreline points (M x 3) and inner radius (M,) for the course."""
    dense = _catmull_rom([(x, z, r) for x, z, r in COURSE], SEG_LEN)
    pts = np.stack([tip0[0] + dense[:, 0], np.full(len(dense), tip0[1]), tip0[2] + dense[:, 1]], axis=1)
    return pts, dense[:, 2]


def add_course(spec, tip0):
    """Add the tube walls, target and floor move to the spec (before compile)."""
    pts, radii = course_centreline(tip0)
    wb = spec.worldbody
    y = np.array([0.0, 1.0, 0.0])
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        t = (b - a) / np.linalg.norm(b - a)
        u = np.cross(t, y)
        u /= np.linalg.norm(u)
        v = np.cross(t, u)
        R = 0.5 * (radii[i] + radii[i + 1])
        half_len = np.linalg.norm(b - a) / 2 + 0.004  # overlap so bends have no gaps
        half_w = math.pi * (R + WALL_T) / N_STAVES * 1.15
        for k in range(N_STAVES):
            ang = 2 * math.pi * k / N_STAVES
            n = math.cos(ang) * u + math.sin(ang) * v
            mat = np.stack([n, np.cross(t, n), t], axis=1)  # box x = radial, z = along tube
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, mat.flatten())
            wb.add_geom(name=f"wall_{i}_{k}", type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=[WALL_T / 2, half_w, half_len], pos=(a + b) / 2 + (R + WALL_T / 2) * n,
                        quat=q, contype=0, conaffinity=LINK_GROUP | DISC_GROUP, rgba=WALL_RGBA,
                        friction=[0.3, 0.005, 0.0001])
    wb.add_site(name="target", pos=pts[-1], size=[0.012, 0, 0], rgba=[0.2, 0.9, 0.3, 0.9])
    for g in spec.geoms:  # discs: collide with walls only
        if g.name and g.name.startswith("disc_"):
            g.contype, g.conaffinity = DISC_GROUP, 0
    floor = spec.geom("ground")
    if floor is not None:
        floor.pos = [0.0, 0.0, tip0[2] - 0.6]


class SimViewCourse(uz.SimViewZ):
    def __init__(self, config=u2.CONFIG):
        # straight-robot tip: mount position minus total length along -z (hanging)
        import json
        c = json.load(open(config))
        L = sum(c["segment_lengths"].values())
        mount = np.array([float(v) for v in c["mounting"]["world_pos"].split()])
        self.tip0 = mount - np.array([0.0, 0.0, L])

        def edit(spec):
            uz._make_mount_mocap(spec)
            for i, b in enumerate(spec.bodies):  # name the disc geoms for the collision setup
                if b.name.startswith("disc_") and b.first_geom() is not None:
                    b.first_geom().name = b.name
            add_course(spec, self.tip0)

        u2.SimView2.__init__(self, config, spec_edit=edit)
        self.mocap_id = self.m.body_mocapid[self.m.body("tdcr_mount").id]
        self.base0 = self.d.mocap_pos[self.mocap_id].copy()
        self.centre, self.radii = course_centreline(self.tip0)
        self.arclen = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(self.centre, axis=0), axis=1))])
        self.wall_ids = {g for g in range(self.m.ngeom) if self.m.geom(g).name.startswith("wall_")}
        self.target = self.m.site("target").id
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 0.62, 90, -8
        self._look = self.tip0.copy()

    def wall_contacts(self):
        """(number of robot-wall contacts, total normal force N)."""
        n, f, buf = 0, 0.0, np.zeros(6)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if c.geom1 in self.wall_ids or c.geom2 in self.wall_ids:
                mujoco.mj_contactForce(self.m, self.d, i, buf)
                n, f = n + 1, f + abs(buf[0])
        return n, f

    def progress(self):
        """Tip arc-length along the course centreline (m) and total length (m)."""
        tip = self.tip_world()
        i = int(np.argmin(np.linalg.norm(self.centre - tip, axis=1)))
        return self.arclen[i], self.arclen[-1]

    def lowest_point(self):
        """Height above the (moved) floor, not world z = 0."""
        return super().lowest_point() - float(self.m.geom_pos[self.m.geom("ground").id][2])

    def target_dist(self):
        return float(np.linalg.norm(self.tip_world() - self.d.site_xpos[self.target]))

    def render(self):
        # camera follows the middle of the robot, smoothed
        mid = 0.5 * (self.tip_world() + self.d.xpos[self.m.body("tdcr_mount").id])
        self._look += 0.15 * (mid - self._look)
        self.cam.lookat[:] = self._look
        return super().render()


class AppCourse(uz.AppZ):
    def make_sim(self, args):
        return SimViewCourse(args.config)

    def __init__(self, args):
        super().__init__(args)
        self.t_start, self.t_done, self.max_force = time.perf_counter(), None, 0.0
        import pygame
        pygame.display.set_caption("2-segment TDCR - obstacle course (sim only)")

    def handle(self, e):
        super().handle(e)
        import pygame
        if e.type == pygame.KEYDOWN and e.key == pygame.K_t:
            self.t_start, self.t_done, self.max_force = time.perf_counter(), None, 0.0

    def hints(self):
        h = super().hints()
        return h[:4] + ["T: reset timer   target = green dot"] + h[4:]

    def status_lines(self):
        lines = super().status_lines()
        n, f = self.sim.wall_contacts()
        self.max_force = max(self.max_force, f)
        s, total = self.sim.progress()
        dist = self.sim.target_dist() * 1000
        if self.t_done is None and dist < 20:
            self.t_done = time.perf_counter() - self.t_start
        t = self.t_done if self.t_done is not None else time.perf_counter() - self.t_start
        done = "  REACHED TARGET" if self.t_done is not None else ""
        course = (f"course {100 * s / total:5.1f}%   target {dist:6.1f} mm   time {t:6.1f} s{done}   "
                  f"wall contacts {n:2d}  force {f:5.2f} N (max {self.max_force:4.2f})",
                  ui.WARN if n else ui.FG)
        return lines[:-1] + [course, lines[-1]]


class _Args(argparse.Namespace):
    pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(u2.CONFIG))
    p.add_argument("--rate", type=float, default=90.0, help="max bend slew (deg/s)")
    p.add_argument("--limit", type=float, default=60.0, help="bend slider range +/- deg")
    p.add_argument("--youngs-mpa", type=float, help="starting backbone E (default: from config)")
    p.add_argument("--e-min", type=float, default=0.5)
    p.add_argument("--e-max", type=float, default=10.0)
    p.add_argument("--z-up", type=float, default=50.0, help="Z slider max (mm up)")
    p.add_argument("--z-down", type=float, default=300.0, help="Z slider max (mm down = insertion)")
    p.add_argument("--z-rate", type=float, default=30.0, help="max Z slew (mm/s)")
    AppCourse(p.parse_args()).run()


if __name__ == "__main__":
    main()
