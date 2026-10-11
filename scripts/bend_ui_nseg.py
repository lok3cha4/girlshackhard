"""N-segment obstacle-course sim: segment count and segment length as sliders.

Builds on bend_ui_2seg_course.py (same course, Z insertion, live stiffness,
gravity toggle, contact/progress/target feedback) but the robot is
generated from two extra sliders:

  * Segments: 1-4. Pitch/yaw sliders appear for each segment.
  * Segment length: 40-200 mm in 10 mm steps.
  * Discs per segment: 1-16, independent of length (pitch = length / discs).
    Limited so neighbouring discs keep at least a 1 mm gap; the status line
    shows pitch, gap, and the bend at which neighbouring disc rims would touch
    (discs don't collide with each other in the sim, so that is a warning).

Changing any of the three rebuilds the model (regenerate + compile, under a
second):
the robot comes back straight, Z homed, outside the course. Bend and
stiffness settings are kept. Everything else matches the earlier sims: 4
tendons per segment at 17 mm, segment s's holes rotated s * 90/N deg so all
4N holes are distinct (2 segments: 0 / 45 deg as before), discs carry 4 holes
per segment they pass, 2 virtual servos per segment (R_SPOOL, 30-150 deg)
drive the sim, coupled routing through proximal segments.

    uv run python scripts/bend_ui_nseg.py [--segments 2 --seg-len 160 --discs 8]
    uv run python scripts/bend_ui_nseg.py --view 640x560     # smaller window
Keys: arrows = segment 1, W/S/A/D = segment 2 (others: sliders),
R / F = Z up / down (shift 10 mm), H = home Z, [ / ] = stiffness,
space = center bends, G = gravity, T = reset timer, Esc = close.
"""

import argparse
import json
import math
import time

import mujoco
import numpy as np
import pygame

import bend_ui as ui
import bend_ui_2seg as u2
import bend_ui_2seg_course as uc
import bend_ui_2seg_z as uz
import serial_bridge as sb
import sim_disc_design as sd
import sweep_2seg as sw

LINK_LEN = 0.020 / 3   # target tube link length, m (as in the 20 mm-pitch model)
MIN_GAP = 0.001        # minimum face-to-face gap between neighbouring discs, m
VIEW_W, VIEW_H = 820, 720  # MuJoCo view; shrunk to fit the screen at startup
PANEL_W = 400
N_STATUS = 6


PANEL_MIN_H = 700         # panel content for 4 segments (8 bend sliders + hints)


def status_top():
    return max(VIEW_H, PANEL_MIN_H)


def win_h():
    return status_top() + 6 + N_STATUS * 17 + 6  # status lines below, full width


def fit_view(view_arg=None):
    """Set the MuJoCo view size (from --view WxH or the default), shrunk to fit the screen."""
    global VIEW_W, VIEW_H
    if view_arg:
        VIEW_W, VIEW_H = (int(v) for v in view_arg.lower().split("x"))
    info = pygame.display.Info()
    if info.current_w > 0 and info.current_h > 0:
        avail_w, avail_h = info.current_w - 40, info.current_h - 90  # dock / menu bar / title
        scale = min(1.0, (avail_w - PANEL_W) / VIEW_W, (avail_h - (win_h() - status_top())) / VIEW_H)
        VIEW_W, VIEW_H = int(VIEW_W * scale) // 2 * 2, int(VIEW_H * scale) // 2 * 2


def max_discs(seg_len):
    return max(1, int(seg_len // (sw.DISC_T / 1000 + MIN_GAP)))
OUT = sb.ROOT / "outputs/nseg"


def make_config(base_config, n_seg, seg_len, discs, course=True):
    """Generation config for n_seg segments of seg_len metres with `discs` discs each.

    Returns (path, discs, links per disc pitch). The tube link count is chosen
    from the pitch so links stay ~6.7 mm whatever the disc count.
    """
    c = json.load(open(base_config))
    discs = min(int(discs), max_discs(seg_len))
    pitch = seg_len / discs
    lpp = max(1, int(round(pitch / LINK_LEN)))
    keys = [str(i + 1) for i in range(n_seg)]
    c.update(
        description=f"{n_seg} segment(s) x {discs} discs at {pitch * 1000:.1f} mm pitch ({seg_len * 1000:g} mm each)",
        num_segments=n_seg,
        links_per_segment={k: discs * lpp for k in keys},
        segment_lengths={k: seg_len for k in keys},
        seg_offsets=[s * (math.pi / 2) / n_seg for s in range(n_seg)],
    )
    c["actuation_details"] = {"segments": [{"number_of_tendons": 4, "distance_to_backbone": sw.TENDON_OFFSET / 1000}] * n_seg}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"nseg_{n_seg}x{int(round(seg_len * 1000))}mm_{discs}d.json"
    json.dump(c, open(path, "w"), indent=2)
    return path, discs, lpp


def servo_angles(segs, bends, discs):
    """2 virtual servos per segment -> (clamped, raw)."""
    dl = sb.tendon_dl(segs, bends, discs)
    raw = []
    for s in range(len(segs)):
        for p, a in ((0, 2), (1, 3)):
            raw.append(sb.REST_DEG + math.degrees((dl[(s, a)] - dl[(s, p)]) / 2 / sb.R_SPOOL))
    return [min(sb.SAFE_MAX, max(sb.SAFE_MIN, v)) for v in raw], raw


class Controller:
    """Slew-limited pitch/yaw per segment (2 axes per segment)."""

    def __init__(self, segs, discs, rate_dps, limit_deg, start=None):
        self.segs, self.discs, self.rate, self.limit = segs, discs, rate_dps, limit_deg
        n = 2 * len(segs)
        self.target, self.current = [0.0] * n, [0.0] * n
        self.angles, self.clamped = [sb.REST_DEG] * n, False

    def set_target(self, values):
        self.target = [max(-self.limit, min(self.limit, float(v))) for v in values]

    def bends(self):
        out = []
        for s in range(len(self.segs)):
            p, y = self.current[2 * s], self.current[2 * s + 1]
            out.append((math.radians(math.hypot(p, y)), math.atan2(y, p) if (p or y) else 0.0))
        return out

    def tick(self, dt):
        step = self.rate * dt
        self.current = [c + max(-step, min(step, t - c)) for c, t in zip(self.current, self.target)]
        self.angles, raw = servo_angles(self.segs, self.bends(), self.discs)
        self.clamped = any(abs(r - c) > 1e-9 for r, c in zip(raw, self.angles))

    def settled(self):
        return all(abs(t - c) < 1e-6 for t, c in zip(self.target, self.current))


class SimViewN:
    """N-segment model: mocap mount (Z), optional course, virtual-servo drive."""

    def __init__(self, cfg_path, discs, n_seg, lpp, course=True):
        c = json.load(open(cfg_path))
        self.E = c["material_properties"]["youngs_modulus"] / 1e6
        self.L = sum(c["segment_lengths"].values())
        self.n_seg, self.discs = n_seg, discs
        mount = np.array([float(v) for v in c["mounting"]["world_pos"].split()])
        self.tip0 = mount - np.array([0.0, 0.0, self.L])
        self.course = course

        def edit(spec):
            uz._make_mount_mocap(spec)
            for b in spec.bodies:
                if b.name.startswith("disc_") and b.first_geom() is not None:
                    b.first_geom().name = b.name
            if course:
                uc.add_course(spec, self.tip0)
            else:  # still keep a long robot off the floor
                spec.geom("ground").pos = [0.0, 0.0, self.tip0[2] - 0.6]

        xml = sw.generate(cfg_path, sw.HANG_EULER, OUT)
        self.m, self.d = sd.build(xml, discs, lpp, True, n_segments=n_seg, spec_edit=edit)
        self.lpp = lpp
        self.pitch = self.L / n_seg / discs
        self.ctrl0 = self.d.ctrl.copy()
        self.act = {}
        for a in range(self.m.nu):
            _, s, _, n = self.m.actuator(a).name.split("_")
            self.act[(int(s), int(n))] = a
        self.mocap_id = self.m.body_mocapid[self.m.body("tdcr_mount").id]
        self.base0 = self.d.mocap_pos[self.mocap_id].copy()
        self.E0, self.stiff0 = self.E, self.m.jnt_stiffness.copy()
        self.wall_ids = {g for g in range(self.m.ngeom) if self.m.geom(g).name.startswith("wall_")}
        if course:
            self.centre, _ = uc.course_centreline(self.tip0)
            self.arclen = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(self.centre, axis=0), axis=1))])
            self.target = self.m.site("target").id
        self.m.vis.global_.offwidth = max(self.m.vis.global_.offwidth, VIEW_W)
        self.m.vis.global_.offheight = max(self.m.vis.global_.offheight, VIEW_H)
        self.renderer = mujoco.Renderer(self.m, height=VIEW_H, width=VIEW_W)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.azimuth, self.cam.elevation = 90, -8
        self.cam.distance = max(0.62, 1.7 * self.L)
        self._look = self.tip0 + np.array([0.0, 0.0, self.L / 2])

    # --- control / physics
    def set_E(self, E_mpa):
        self.E = E_mpa
        self.m.jnt_stiffness[:] = self.stiff0 * (E_mpa / self.E0)

    def set_z(self, z_mm):
        self.d.mocap_pos[self.mocap_id] = self.base0 + np.array([0.0, 0.0, z_mm / 1000.0])

    def apply(self, angles):
        u = self.ctrl0.copy()
        for i, ang in enumerate(angles):
            s, (pr, an) = i // 2, ((0, 2), (1, 3))[i % 2]
            pull = math.radians(round(ang) - sb.REST_DEG) * sb.R_SPOOL
            u[self.act[(s, pr)]] -= pull
            u[self.act[(s, an)]] += pull
        self.d.ctrl[:] = u

    def step(self, dt):
        for _ in range(min(50, max(1, int(round(dt / self.m.opt.timestep))))):
            mujoco.mj_step(self.m, self.d)

    # --- measurements
    def tip_world(self):
        return self.d.xpos[self.m.body("EE_pos").id].copy()

    def tip_pos(self):
        o, R = sw.mount_frame(self.m, self.d)
        return R.T @ (self.tip_world() - o)

    def tip_bend(self):
        _, R = sw.mount_frame(self.m, self.d)
        z = R.T @ self.d.xmat[self.m.body("EE_pos").id].reshape(3, 3)[:, 2]
        return math.degrees(math.acos(max(-1.0, min(1.0, z[2]))))

    def wall_contacts(self):
        n, f, buf = 0, 0.0, np.zeros(6)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if c.geom1 in self.wall_ids or c.geom2 in self.wall_ids:
                mujoco.mj_contactForce(self.m, self.d, i, buf)
                n, f = n + 1, f + abs(buf[0])
        return n, f

    def progress(self):
        i = int(np.argmin(np.linalg.norm(self.centre - self.tip_world(), axis=1)))
        return self.arclen[i], self.arclen[-1]

    def target_dist(self):
        return float(np.linalg.norm(self.tip_world() - self.d.site_xpos[self.target]))

    def render(self):
        mid = 0.5 * (self.tip_world() + self.d.xpos[self.m.body("tdcr_mount").id])
        self._look += 0.15 * (mid - self._look)
        self.cam.lookat[:] = self._look
        self.renderer.update_scene(self.d, self.cam)
        # shadows + floor reflections cost 2.6-4x render time with ~500-1000 geoms
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        return self.renderer.render()

    def close(self):
        self.renderer.close()


class App:
    def __init__(self, args):
        pygame.init()
        self.args = args
        fit_view(args.view)
        self.screen = pygame.display.set_mode((VIEW_W + PANEL_W, win_h()))
        pygame.display.set_caption("N-segment TDCR - obstacle course (sim only)")
        self.font = pygame.font.SysFont("menlo,monaco,courier", 14)
        self.small = pygame.font.SysFont("menlo,monaco,courier", 12)
        x0, self.w = VIEW_W + 20, PANEL_W - 40
        self.x0 = x0
        self.nseg_s = ui.Slider("Segments", (x0, 38, self.w, 10), 1, 4, unit="", step=1, fmt="{:.0f}")
        self.len_s = ui.Slider("Seg length", (x0, 80, self.w, 10), 40, 200, unit="mm", step=10, fmt="{:.0f}")
        self.disc_s = ui.Slider("Discs/seg", (x0, 122, self.w, 10), 1, 16, unit="", step=1, fmt="{:.0f}")
        self.stiff = ui.Slider("Stiffness E", (x0, 164, self.w, 10), args.e_min, args.e_max,
                               unit="MPa", step=0.1, fmt="{:5.1f}")
        self.zs = ui.Slider("Z (up/down)", (x0, 206, self.w, 10), -args.z_down, args.z_up,
                            unit="mm", step=1, fmt="{:+5.0f}")
        self.nseg_s.value, self.len_s.value, self.disc_s.value = args.segments, args.seg_len, args.discs
        self.bend_vals = [0.0] * 8  # remembered across rebuilds (max 4 segments)
        self.sim = None
        self.closing, self.fps, self.msg = False, 0.0, ""
        self.rebuild(first=True)
        self.stiff.value = round(self.sim.E if args.youngs_mpa is None else args.youngs_mpa, 1)

    # ---- model (re)build
    def rebuild(self, first=False):
        n, L = int(self.nseg_s.value), self.len_s.value / 1000
        if self.sim is not None:
            self.bend_vals[:len(self.sliders)] = [s.value for s in self.sliders]
            self.sim.close()
        t0 = time.perf_counter()
        cfg, discs, lpp = make_config(self.args.config, n, L, self.disc_s.value, not self.args.no_course)
        self.sim = SimViewN(cfg, discs, n, lpp, course=not self.args.no_course)
        if not first:
            self.sim.set_E(self.stiff.value)
        self.built = self.wanted()
        self.g0 = self.sim.m.opt.gravity.copy()
        self.segs = sb.load_layout(cfg)
        self.ctrl = Controller(self.segs, discs, self.args.rate, self.args.limit)
        y = 256
        self.sliders = []
        for s in range(n):
            for k, axis in enumerate(("pitch", "yaw  ")):
                sl = ui.Slider(f"Seg {s + 1} {axis}", (self.x0, y, self.w, 10), -self.args.limit, self.args.limit)
                sl.value = self.bend_vals[2 * s + k]
                self.sliders.append(sl)
                y += 40
        self.btn_center = pygame.Rect(self.x0, y - 8, 90, 24)
        self.hint_y = y + 26
        self.zs.value, self.z_cur = 0.0, 0.0
        self.t_start, self.t_done, self.max_force = time.perf_counter(), None, 0.0
        capped = f" (capped from {int(self.disc_s.value)}: needs a {MIN_GAP * 1000:g} mm gap)" if discs < self.disc_s.value else ""
        self.msg = (f"built {n} x {discs} discs{capped}, {n * L * 1000:g} mm, {self.sim.m.nbody} bodies "
                    f"in {time.perf_counter() - t0:.1f} s")

    # ---- input
    def widgets(self):
        return [self.nseg_s, self.len_s, self.disc_s, self.stiff, self.zs] + self.sliders

    def wanted(self):
        return int(self.nseg_s.value), self.len_s.value, int(self.disc_s.value)

    def handle(self, e):
        if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
            self.closing = True
            self.center()
            return
        if self.closing:
            return
        for s in self.widgets():
            s.handle(e)
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and self.btn_center.collidepoint(e.pos):
            self.center()
        if e.type != pygame.KEYDOWN:
            return
        step = 5 if e.mod & pygame.KMOD_SHIFT else 1
        keys = {pygame.K_UP: (0, step), pygame.K_DOWN: (0, -step), pygame.K_RIGHT: (1, step),
                pygame.K_LEFT: (1, -step), pygame.K_w: (2, step), pygame.K_s: (2, -step),
                pygame.K_d: (3, step), pygame.K_a: (3, -step)}
        if e.key in keys and keys[e.key][0] < len(self.sliders):
            i, dv = keys[e.key]
            sl = self.sliders[i]
            sl.value = max(sl.lo, min(sl.hi, sl.value + dv))
        elif e.key in (pygame.K_r, pygame.K_f):
            dz = (10 if e.mod & pygame.KMOD_SHIFT else 1) * (1 if e.key == pygame.K_r else -1)
            self.zs.value = max(self.zs.lo, min(self.zs.hi, self.zs.value + dz))
        elif e.key == pygame.K_h:
            self.zs.value = 0.0
        elif e.key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
            dv = 0.1 if e.key == pygame.K_RIGHTBRACKET else -0.1
            self.stiff.value = round(max(self.stiff.lo, min(self.stiff.hi, self.stiff.value + dv)), 1)
        elif e.key == pygame.K_SPACE:
            self.center()
        elif e.key == pygame.K_g:
            g = self.sim.m.opt.gravity
            g[:] = 0 if g.any() else self.g0
        elif e.key == pygame.K_t:
            self.t_start, self.t_done, self.max_force = time.perf_counter(), None, 0.0

    def center(self):
        for s in self.sliders:
            s.value = 0.0

    # ---- frame
    def frame(self, dt):
        dragging = self.nseg_s.drag or self.len_s.drag or self.disc_s.drag
        if self.wanted() != self.built and not dragging and not self.closing:
            self.rebuild()
        if abs(self.stiff.value - self.sim.E) > 1e-9:
            self.sim.set_E(self.stiff.value)
        step = self.args.z_rate * dt
        self.z_cur += max(-step, min(step, self.zs.value - self.z_cur))
        self.sim.set_z(self.z_cur)
        self.ctrl.set_target([s.value for s in self.sliders])
        self.ctrl.tick(dt)
        self.sim.apply(self.ctrl.angles)
        self.sim.step(dt)
        rgb = self.sim.render()
        if self.closing and self.ctrl.settled():
            return False
        self.draw(rgb)
        return True

    def status_lines(self):
        sim, b = self.sim, self.ctrl.bends()
        err = np.linalg.norm(sim.tip_pos() - u2.cc_tip(self.segs, b)) * 1000
        cmd = "  ".join(f"s{i + 1} {math.degrees(t):4.1f}@{math.degrees(p):+4.0f}" for i, (t, p) in enumerate(b))
        a = self.ctrl.angles
        lines = [
            (f"cmd {cmd}   (E {sim.E:.1f} MPa, {sim.n_seg} x {sim.discs} discs, {sim.L * 1000:g} mm)", ui.FG),
            (f"sim tip {sim.tip_bend():5.1f} deg   tip vs CC {err:5.1f} mm   base Z {self.z_cur:+6.1f} mm", ui.FG),
            self.disc_line(b),
            ("virtual servos " + " ".join(f"{round(v):3d}" for v in a) +
             ("  CLAMPED" if self.ctrl.clamped else "") + f"   {self.fps:4.1f} fps",
             ui.WARN if self.ctrl.clamped else ui.FG),
        ]
        if sim.course:
            n, f = sim.wall_contacts()
            self.max_force = max(self.max_force, f)
            s, tot = sim.progress()
            dist = sim.target_dist() * 1000
            if self.t_done is None and dist < 20:
                self.t_done = time.perf_counter() - self.t_start
            t = self.t_done if self.t_done is not None else time.perf_counter() - self.t_start
            lines.append((f"course {100 * s / tot:5.1f}%  target {dist:6.1f} mm  time {t:6.1f} s"
                          + ("  REACHED" if self.t_done is not None else "")
                          + f"  contacts {n:2d}  force {f:5.2f} N (max {self.max_force:4.2f})",
                          ui.WARN if n else ui.FG))
        lines.append((("closing: ramping to straight...  " if self.closing else "") + self.msg, ui.DIM))
        return lines

    def disc_line(self, bends):
        """Pitch, gap, and the per-segment bend at which neighbouring disc rims touch."""
        sim = self.sim
        gap = sim.pitch - sw.DISC_T / 1000
        rim = math.degrees(sim.discs * gap / (sw.DISC_OD / 2000))  # closure R*theta/N = gap
        worst = max(math.degrees(t) for t, _ in bends)
        over = worst > rim
        return (f"discs {sim.discs}/seg  pitch {sim.pitch * 1000:5.1f} mm  gap {gap * 1000:5.1f} mm  "
                f"rims touch at {rim:5.0f} deg/seg" + (f"  -> commanded {worst:.0f} deg: DISCS WOULD COLLIDE" if over else "")
                + f"   ({sim.lpp} links/pitch)", ui.WARN if over else ui.FG)

    def draw(self, rgb):
        scr = self.screen
        scr.fill(ui.BG)
        scr.blit(pygame.image.frombuffer(rgb, (VIEW_W, VIEW_H), "RGB"), (0, 0))
        for s in self.widgets():
            s.draw(scr, self.font)
        pending = self.wanted() != self.built
        if pending:
            scr.blit(self.small.render("release to rebuild", True, ui.WARN), (self.x0 + 200, 16))
        pygame.draw.rect(scr, (60, 62, 70), self.btn_center, border_radius=4)
        scr.blit(self.font.render("Center", True, ui.FG), (self.btn_center.x + 12, self.btn_center.y + 4))
        hints = ["arrows: seg 1  W/S/A/D: seg 2  space: center", "R/F: Z up/down (shift 10)  H: home Z",
                 "[ ]: stiffness  T: timer  G: gravity " + ("ON" if self.sim.m.opt.gravity.any() else "OFF"),
                 "segments/length/discs: rebuild on release", "sim only - nothing is sent"]
        for i, t in enumerate(hints):
            scr.blit(self.small.render(t, True, ui.DIM), (self.x0, self.hint_y + i * 17))
        for i, (t, c) in enumerate(self.status_lines()):
            scr.blit(self.small.render(t, True, c), (10, status_top() + 6 + i * 17))
        if self.disc_s.value > max_discs(self.len_s.value / 1000):
            scr.blit(self.small.render(f"max {max_discs(self.len_s.value / 1000)} at this length", True, ui.WARN),
                     (self.x0 + 200, self.disc_s.rect.y - 22))
        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()
        dt, t_fps, n = 1.0 / ui.TICK_HZ, time.perf_counter(), 0
        try:
            while True:
                for e in pygame.event.get():
                    self.handle(e)
                if not self.frame(dt):
                    break
                dt = min(clock.tick(ui.TICK_HZ) / 1000.0, 0.05)  # cap after a rebuild pause
                n += 1
                now = time.perf_counter()
                if now - t_fps >= 1.0:
                    self.fps, t_fps, n = n / (now - t_fps), now, 0
        finally:
            self.sim.close()
            pygame.quit()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(u2.CONFIG), help="base config (materials, discs, mount)")
    p.add_argument("--segments", type=int, default=2, choices=[1, 2, 3, 4])
    p.add_argument("--seg-len", type=float, default=160.0, help="segment length, mm (40-200, steps of 10)")
    p.add_argument("--discs", type=int, default=8, help="discs per segment (1-16)")
    p.add_argument("--view", help="MuJoCo view size WxH (default 820x720, shrunk to fit the screen)")
    p.add_argument("--no-course", action="store_true", help="free space instead of the obstacle course")
    p.add_argument("--rate", type=float, default=90.0, help="max bend slew (deg/s)")
    p.add_argument("--limit", type=float, default=60.0, help="bend slider range +/- deg")
    p.add_argument("--youngs-mpa", type=float, help="starting backbone E (default: from config)")
    p.add_argument("--e-min", type=float, default=0.5)
    p.add_argument("--e-max", type=float, default=10.0)
    p.add_argument("--z-up", type=float, default=50.0)
    p.add_argument("--z-down", type=float, default=300.0)
    p.add_argument("--z-rate", type=float, default=30.0, help="max Z slew (mm/s)")
    App(p.parse_args()).run()


if __name__ == "__main__":
    main()
