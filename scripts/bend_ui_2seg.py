"""Sim-only 2-segment pitch/yaw UI (pygame) with a live MuJoCo-predicted shape.

Same control style as bend_ui.py, for configs/generation/two_segment_8disc.json
(2 x 8 discs at 20 mm pitch, 2 x 160 mm, segment-2 holes rotated 45 deg):
four drag sliders (segment 1 pitch/yaw, segment 2 pitch/yaw), Center, and an
offscreen mujoco.Renderer view (hanging, gravity, real PLA disc mass, tendons
routed through the disc holes).

Pitch/yaw are in each segment's own base frame (0 deg = toward segment 1's
A0 hole, 90 = toward its A1 hole; segment 2's frame is the end of segment 1).
Each tick the commands are slew-limited (--rate deg/s), turned into tendon
lengths with serial_bridge.tendon_dl (coupled routing: segment-2 tendons run
through segment 1), mapped onto 4 *virtual* servos (one per antagonistic
pair, R_SPOOL, 30-150 deg, integer degrees -- what a 4-servo build would
send), and the sim's tendons are driven by those servo angles, so clamping
shows up in the predicted shape. Nothing is sent over serial.

    uv run python scripts/bend_ui_2seg.py
A fifth slider sets the backbone Young's modulus E live (0.5-10 MPa). In the
generator's material mode every joint stiffness is proportional to E (checked:
regenerating at another E scales all 96 joint stiffnesses by exactly E/E0 and
changes nothing else), so the slider rescales the compiled joint stiffness
instead of rebuilding the model.

Keys: arrows = segment 1 (up/down pitch, left/right yaw), W/S/A/D = segment 2,
shift = 5 deg steps, [ / ] = stiffness -/+ 0.1 MPa, space = center (bends only),
G = toggle gravity, Esc = close.
"""

import argparse
import json
import math
import time

import mujoco
import numpy as np
import pygame

import bend_ui as ui
import serial_bridge as sb
import sim_disc_design as sd
import sweep_2seg as sw

CONFIG = sb.ROOT / "configs/generation/two_segment_8disc.json"
OUT = sb.ROOT / "outputs/two_seg"
DISCS, LPP = 8, 3
# virtual 4-servo build: (segment, primary, antagonist) per servo A0..A3
PAIRS = [(0, 0, 2), (0, 1, 3), (1, 0, 2), (1, 1, 3)]
VIEW_W, VIEW_H = 560, 560


def servo_angles(segs, bends):
    """bends [(theta, phi)] rad per segment -> (clamped, raw) for the 4 virtual servos."""
    dl = sb.tendon_dl(segs, bends, DISCS)
    raw = [sb.REST_DEG + math.degrees((dl[(s, a)] - dl[(s, p)]) / 2 / sb.R_SPOOL) for s, p, a in PAIRS]
    return [min(sb.SAFE_MAX, max(sb.SAFE_MIN, v)) for v in raw], raw


def cc_tip(segs, bends):
    """Composed constant-curvature tip position (m) in the robot base frame."""
    pos, R = np.zeros(3), np.eye(3)
    for seg, (th, phi) in zip(segs, bends):
        L = seg["L"]
        if abs(th) < 1e-9:
            p_loc, R_loc = np.array([0.0, 0.0, L]), np.eye(3)
        else:
            r = L / th
            p_loc = np.array([r * (1 - math.cos(th)) * math.cos(phi), r * (1 - math.cos(th)) * math.sin(phi),
                              r * math.sin(th)])
            k = np.array([-math.sin(phi), math.cos(phi), 0.0])  # bend axis
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            R_loc = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K
        pos, R = pos + R @ p_loc, R @ R_loc
    return pos


class Controller:
    """Slew-limited 4-axis command (seg1 pitch/yaw, seg2 pitch/yaw)."""

    def __init__(self, segs, rate_dps=90.0, limit_deg=60.0):
        self.segs, self.rate, self.limit = segs, rate_dps, limit_deg
        self.target, self.current = [0.0] * 4, [0.0] * 4
        self.angles, self.clamped = [sb.REST_DEG] * 4, False

    def set_target(self, values):
        self.target = [max(-self.limit, min(self.limit, float(v))) for v in values]

    def bends(self):
        """[(theta, phi)] in radians per segment from the current pitch/yaw."""
        out = []
        for s in range(2):
            p, y = self.current[2 * s], self.current[2 * s + 1]
            out.append((math.radians(math.hypot(p, y)), math.atan2(y, p) if (p or y) else 0.0))
        return out

    def tick(self, dt):
        step = self.rate * dt
        self.current = [c + max(-step, min(step, t - c)) for c, t in zip(self.current, self.target)]
        self.angles, raw = servo_angles(self.segs, self.bends())
        self.clamped = any(abs(r - c) > 1e-9 for r, c in zip(raw, self.angles))

    def settled(self):
        return all(abs(t - c) < 1e-6 for t, c in zip(self.target, self.current))


class SimView2:
    """2-segment model (sim_disc_design.build), driven by the virtual servo angles."""

    def __init__(self, config=CONFIG):
        c = json.load(open(config))
        self.E = c["material_properties"]["youngs_modulus"] / 1e6
        self.L = sum(c["segment_lengths"].values())
        OUT.mkdir(parents=True, exist_ok=True)
        cfg = OUT / "two_segment_8disc.json"
        json.dump(c, open(cfg, "w"), indent=2)
        xml = sw.generate(cfg, sw.HANG_EULER, OUT)
        self.m, self.d = sd.build(xml, DISCS, LPP, True, n_segments=2)
        self.ctrl0 = self.d.ctrl.copy()
        self.act = {}
        for a in range(self.m.nu):
            _, s, _, n = self.m.actuator(a).name.split("_")  # seg_<s>_ten_<n>
            self.act[(int(s), int(n))] = a
        self.seg1_end = self.m.body(f"link_{DISCS * LPP}").id
        self.E0, self.stiff0 = self.E, self.m.jnt_stiffness.copy()
        self.m.vis.global_.offwidth = max(self.m.vis.global_.offwidth, VIEW_W)
        self.m.vis.global_.offheight = max(self.m.vis.global_.offheight, VIEW_H)
        self.renderer = mujoco.Renderer(self.m, height=VIEW_H, width=VIEW_W)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        o, _ = sw.mount_frame(self.m, self.d)
        self.cam.lookat[:] = o + np.array([0.0, 0.0, -self.L * 0.55])
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 2.6 * self.L, 135, -12

    def apply(self, angles):
        u = self.ctrl0.copy()
        for (s, pr, an), ang in zip(PAIRS, angles):
            pull = math.radians(round(ang) - sb.REST_DEG) * sb.R_SPOOL
            u[self.act[(s, pr)]] -= pull
            u[self.act[(s, an)]] += pull
        self.d.ctrl[:] = u

    def set_E(self, E_mpa):
        """Backbone Young's modulus (MPa): joint stiffness scales linearly with E."""
        self.E = E_mpa
        self.m.jnt_stiffness[:] = self.stiff0 * (E_mpa / self.E0)

    def step(self, dt):
        for _ in range(min(50, max(1, int(round(dt / self.m.opt.timestep))))):
            mujoco.mj_step(self.m, self.d)

    def _axis_angle(self, body):
        _, R = sw.mount_frame(self.m, self.d)
        z = R.T @ self.d.xmat[body].reshape(3, 3)[:, 2]
        return math.degrees(math.acos(max(-1.0, min(1.0, z[2])))), math.degrees(math.atan2(z[1], z[0]))

    def seg1_bend(self):
        return self._axis_angle(self.seg1_end)

    def tip_bend(self):
        return self._axis_angle(self.m.body("EE_pos").id)

    def tip_pos(self):
        o, R = sw.mount_frame(self.m, self.d)
        return R.T @ (self.d.xpos[self.m.body("EE_pos").id] - o)

    def render(self):
        self.renderer.update_scene(self.d, self.cam)
        return self.renderer.render()

    def close(self):
        self.renderer.close()


class App:
    def __init__(self, args):
        pygame.init()
        self.screen = pygame.display.set_mode((VIEW_W + ui.PANEL_W, VIEW_H + 90))
        pygame.display.set_caption("2-segment TDCR pitch / yaw - live sim (no board)")
        self.font = pygame.font.SysFont("menlo,monaco,courier", 14)
        self.sim = SimView2(args.config)
        self.segs = sb.load_layout(args.config)
        self.ctrl = Controller(self.segs, args.rate, args.limit)
        x0, w = VIEW_W + 20, ui.PANEL_W - 40
        names = ["Seg 1 pitch", "Seg 1 yaw  ", "Seg 2 pitch", "Seg 2 yaw  "]
        ys = [50, 105, 185, 240]
        self.sliders = [ui.Slider(n, (x0, y, w, 10), -args.limit, args.limit) for n, y in zip(names, ys)]
        self.stiff = ui.Slider("Stiffness E", (x0, 320, w, 10), args.e_min, args.e_max,
                               unit="MPa", step=0.1, fmt="{:5.1f}")
        if args.youngs_mpa is not None:
            self.sim.set_E(args.youngs_mpa)
        self.stiff.value = round(self.sim.E, 1)
        self.btn_center = pygame.Rect(x0, 350, 90, 26)
        self.closing, self.fps = False, 0.0
        self.g0 = self.sim.m.opt.gravity.copy()

    def handle(self, e):
        if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
            self.closing = True
            self.center()
            return
        if self.closing:
            return
        for s in self.sliders + [self.stiff]:
            s.handle(e)
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and self.btn_center.collidepoint(e.pos):
            self.center()
        if e.type == pygame.KEYDOWN:
            step = 5 if e.mod & pygame.KMOD_SHIFT else 1
            keys = {pygame.K_UP: (0, step), pygame.K_DOWN: (0, -step), pygame.K_RIGHT: (1, step),
                    pygame.K_LEFT: (1, -step), pygame.K_w: (2, step), pygame.K_s: (2, -step),
                    pygame.K_d: (3, step), pygame.K_a: (3, -step)}
            if e.key in keys:
                i, dv = keys[e.key]
                s = self.sliders[i]
                s.value = max(s.lo, min(s.hi, s.value + dv))
            elif e.key == pygame.K_SPACE:
                self.center()
            elif e.key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
                dv = 0.1 if e.key == pygame.K_RIGHTBRACKET else -0.1
                self.stiff.value = round(max(self.stiff.lo, min(self.stiff.hi, self.stiff.value + dv)), 1)
            elif e.key == pygame.K_g:
                g = self.sim.m.opt.gravity
                g[:] = 0 if g.any() else self.g0

    def center(self):
        for s in self.sliders:
            s.value = 0.0

    def frame(self, dt):
        if abs(self.stiff.value - self.sim.E) > 1e-9:
            self.sim.set_E(self.stiff.value)
        self.ctrl.set_target([s.value for s in self.sliders])
        self.ctrl.tick(dt)
        self.sim.apply(self.ctrl.angles)
        self.sim.step(dt)
        rgb = self.sim.render()
        if self.closing and self.ctrl.settled():
            return False
        self.draw(rgb)
        return True

    def draw(self, rgb):
        scr = self.screen
        scr.fill(ui.BG)
        scr.blit(pygame.image.frombuffer(rgb, (VIEW_W, VIEW_H), "RGB"), (0, 0))
        for s in self.sliders + [self.stiff]:
            s.draw(scr, self.font)
        pygame.draw.rect(scr, (60, 62, 70), self.btn_center, border_radius=4)
        scr.blit(self.font.render("Center", True, ui.FG), (self.btn_center.x + 12, self.btn_center.y + 5))
        hint = ["arrows: seg 1   W/S/A/D: seg 2", "shift: 5 deg   space: center", "[ / ]: stiffness -/+ 0.1 MPa",
                "G: gravity " + ("ON" if self.sim.m.opt.gravity.any() else "OFF (CC check)"),
                "sim only - nothing is sent"]
        for i, t in enumerate(hint):
            scr.blit(self.font.render(t, True, ui.DIM), (VIEW_W + 20, 400 + i * 20))

        b = self.ctrl.bends()
        s1t, s1p = self.sim.seg1_bend()
        tt, tp = self.sim.tip_bend()
        err = np.linalg.norm(self.sim.tip_pos() - cc_tip(self.segs, b)) * 1000
        fmt_dir = lambda t, p: f"{t:5.1f} @ " + (f"{p:+6.1f}" if t > 0.5 else "   -- ")
        a = self.ctrl.angles
        lines = [
            (f"cmd seg1 {math.degrees(b[0][0]):5.1f} @ {math.degrees(b[0][1]):+6.1f}   seg2 {math.degrees(b[1][0]):5.1f} @ "
             f"{math.degrees(b[1][1]):+6.1f}   (E {self.sim.E:g} MPa)", ui.FG),
            (f"sim  seg1 end {fmt_dir(s1t, s1p)}   tip {fmt_dir(tt, tp)}   tip vs CC {err:5.1f} mm", ui.FG),
            ("virtual servos " + "  ".join(f"A{i} {round(v):3d}" for i, v in enumerate(a)) +
             ("   CLAMPED" if self.ctrl.clamped else "") + f"   {self.fps:4.1f} fps",
             ui.WARN if self.ctrl.clamped else ui.FG),
            ("closing: ramping to straight..." if self.closing else "", ui.DIM),
        ]
        for i, (t, c) in enumerate(lines):
            scr.blit(self.font.render(t, True, c), (10, VIEW_H + 6 + i * 20))
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
                dt = clock.tick(ui.TICK_HZ) / 1000.0
                n += 1
                now = time.perf_counter()
                if now - t_fps >= 1.0:
                    self.fps, t_fps, n = n / (now - t_fps), now, 0
        finally:
            self.sim.close()
            pygame.quit()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(CONFIG))
    p.add_argument("--rate", type=float, default=90.0, help="max command slew (deg/s)")
    p.add_argument("--limit", type=float, default=60.0, help="slider range +/- deg")
    p.add_argument("--youngs-mpa", type=float, help="starting backbone E (default: from config)")
    p.add_argument("--e-min", type=float, default=0.5, help="stiffness slider min (MPa)")
    p.add_argument("--e-max", type=float, default=10.0, help="stiffness slider max (MPa)")
    App(p.parse_args()).run()


if __name__ == "__main__":
    main()
