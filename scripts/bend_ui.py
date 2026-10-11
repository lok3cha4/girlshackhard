"""Real-time pitch/yaw UI (pygame) with a live MuJoCo-predicted shape.

One window, one event loop: two drag sliders (pitch, yaw), Center, a serial
port selector, a status line, and an offscreen mujoco.Renderer view of the
1-segment, 8-disc model (hanging, gravity, real disc mass) blitted each frame.

Pitch bends toward A0's primary tendon hole (0 deg), yaw toward A1's (90 deg);
together they form one constant-curvature bend theta = hypot(pitch, yaw) in
direction phi = atan2(yaw, pitch). 2 servos: A0 = pitch pair, A1 = yaw pair.

Each 50 Hz tick: slew the command toward the sliders (--rate deg/s), compute
servo angles with serial_bridge.servo_angles, drive the sim's tendons with
exactly those (clamped, integer) angles x R_SPOOL, step physics in real time
(mj_forward alone would not move the robot), render, and -- only if a port
is open -- send the line when the integer angles change. Closing the window
ramps back to straight (sending the rest line if a board is connected) and
closes the port.

The port list defaults to "No board / sim only" (nothing opened or sent).
Picking a port opens it in the background (2 s reset wait) without freezing
the UI; open/write failures are shown and fall back to "No board".

    uv run python scripts/bend_ui.py                      # sim only
    uv run python scripts/bend_ui.py --port /dev/cu.usbmodemXXXX
    uv run python scripts/bend_ui.py --no-serial          # sim only + echo lines
A third slider sets the backbone Young's modulus E live (0.5-10 MPa). Joint
stiffness from the generator's material mode is proportional to E (checked:
regenerating at another E scales all joint stiffnesses by exactly E/E0 and
changes nothing else), so the slider rescales it without rebuilding the model.

Keys: arrows nudge 1 deg (shift: 5), [ / ] = stiffness -/+ 0.1 MPa,
space = center (bends only), Esc = close.
"""

import argparse
import json
import math
import threading
import time

import mujoco
import numpy as np
import pygame

import one_seg_check as osc
import serial_bridge as sb
import sim_disc_design as sd
import sweep_2seg as sw

TICK_HZ = 50
NO_BOARD = "No board / sim only"
VIEW_W, VIEW_H = 520, 440
PANEL_W = 380
BG, FG, DIM, ACCENT, WARN = (24, 26, 30), (230, 230, 230), (140, 140, 150), (90, 170, 255), (255, 120, 90)


class BendController:
    """Slew-limited pitch/yaw command -> servo angles + line (no UI, no I/O)."""

    def __init__(self, segs, discs, rate_dps=90.0, limit_deg=60.0):
        self.segs, self.discs, self.rate, self.limit = segs, discs, rate_dps, limit_deg
        self.target = [0.0, 0.0]  # pitch, yaw (deg)
        self.current = [0.0, 0.0]
        self.angles = [sb.REST_DEG] * len(sb.SERVO_MAP)
        self.clamped = False
        self.line = sb.format_line(self.angles)

    def set_target(self, pitch=None, yaw=None):
        for i, v in enumerate((pitch, yaw)):
            if v is not None:
                self.target[i] = max(-self.limit, min(self.limit, float(v)))

    def bend(self):
        p, y = self.current
        return math.hypot(p, y), math.degrees(math.atan2(y, p)) if (p or y) else 0.0

    def tick(self, dt):
        step = self.rate * dt
        for i in range(2):
            err = self.target[i] - self.current[i]
            self.current[i] += max(-step, min(step, err))
        theta, phi = self.bend()
        angles, raw, _ = sb.servo_angles(self.segs, math.radians(theta), math.radians(phi), self.discs)
        self.angles = angles
        self.clamped = any(abs(r - c) > 1e-9 for r, c in zip(raw, angles))
        self.line = sb.format_line(angles)

    def settled(self):
        return all(abs(t - c) < 1e-6 for t, c in zip(self.target, self.current))


class SimView:
    """Model from one_seg_check / sim_disc_design, driven by the servo angles."""

    def __init__(self, config=sb.DEFAULT_CONFIG, discs=sb.DEFAULT_DISCS):
        E = json.load(open(config))["material_properties"]["youngs_modulus"] / 1e6
        xml, self.L = osc.make_xml(E)
        self.m, self.d = sd.build(xml, discs, osc.LPP, True, n_segments=1)
        self.ctrl0 = self.d.ctrl.copy()
        self.act = {int(self.m.actuator(a).name.split("_")[3]): a for a in range(self.m.nu)}
        self.m.vis.global_.offwidth = max(self.m.vis.global_.offwidth, VIEW_W)
        self.m.vis.global_.offheight = max(self.m.vis.global_.offheight, VIEW_H)
        self.renderer = mujoco.Renderer(self.m, height=VIEW_H, width=VIEW_W)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        o, _ = sw.mount_frame(self.m, self.d)
        self.cam.lookat[:] = o + np.array([0.0, 0.0, -self.L * 0.6])
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 3.6 * self.L, 135, -12
        self.E = E
        self.E0, self.stiff0 = E, self.m.jnt_stiffness.copy()

    def set_E(self, E_mpa):
        """Backbone Young's modulus (MPa): joint stiffness scales linearly with E."""
        self.E = E_mpa
        self.m.jnt_stiffness[:] = self.stiff0 * (E_mpa / self.E0)

    def apply(self, angles):
        """Tendon ctrl = what the spools do at these (sent) servo angles."""
        u = self.ctrl0.copy()
        for (pr, an), sign, ang in zip(sb.SERVO_MAP, sb.SERVO_SIGN, angles):
            pull = sign * math.radians(round(ang) - sb.REST_DEG) * sb.R_SPOOL
            u[self.act[pr]] -= pull
            u[self.act[an]] += pull
        self.d.ctrl[:] = u

    def step(self, dt):
        for _ in range(min(50, max(1, int(round(dt / self.m.opt.timestep))))):
            mujoco.mj_step(self.m, self.d)

    def tip_bend(self):
        _, R = sw.mount_frame(self.m, self.d)
        z = R.T @ self.d.xmat[self.m.body("EE_pos").id].reshape(3, 3)[:, 2]
        return math.degrees(math.acos(max(-1.0, min(1.0, z[2])))), math.degrees(math.atan2(z[1], z[0]))

    def render(self):
        self.renderer.update_scene(self.d, self.cam)
        return self.renderer.render()

    def close(self):
        self.renderer.close()


class Slider:
    def __init__(self, label, rect, lo, hi, unit="deg", step=0.5, fmt="{:+6.1f}"):
        self.label, self.rect, self.lo, self.hi = label, pygame.Rect(rect), lo, hi
        self.unit, self.step, self.fmt = unit, step, fmt
        self.value, self.drag = 0.0, False

    def set_from_x(self, x):
        f = min(1.0, max(0.0, (x - self.rect.x) / self.rect.w))
        self.value = round(round((self.lo + f * (self.hi - self.lo)) / self.step) * self.step, 6)

    def handle(self, e):
        # inflated so the whole knob (radius 9) is grabbable, even at either end
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and self.rect.inflate(22, 16).collidepoint(e.pos):
            self.drag = True
            self.set_from_x(e.pos[0])
        elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
            self.drag = False
        elif e.type == pygame.MOUSEMOTION and self.drag:
            self.set_from_x(e.pos[0])

    def draw(self, surf, font):
        text = f"{self.label} {self.fmt.format(self.value)} {self.unit}"
        surf.blit(font.render(text, True, FG), (self.rect.x, self.rect.y - 22))
        pygame.draw.rect(surf, (60, 62, 70), self.rect, border_radius=4)
        if self.lo < 0 < self.hi:  # zero tick
            cx = self.rect.x + (0 - self.lo) / (self.hi - self.lo) * self.rect.w
            pygame.draw.line(surf, DIM, (cx, self.rect.y - 3), (cx, self.rect.bottom + 3))
        kx = self.rect.x + (self.value - self.lo) / (self.hi - self.lo) * self.rect.w
        pygame.draw.circle(surf, ACCENT, (int(kx), self.rect.centery), 9)


class App:
    def __init__(self, args):
        pygame.init()
        self.screen = pygame.display.set_mode((VIEW_W + PANEL_W, VIEW_H + 70))
        pygame.display.set_caption("TDCR pitch / yaw - live sim")
        self.font = pygame.font.SysFont("menlo,monaco,courier", 14)
        self.small = pygame.font.SysFont("menlo,monaco,courier", 12)
        self.sim = SimView(args.config, args.discs)
        self.ctrl = BendController(sb.load_layout(args.config), args.discs, args.rate, args.limit)
        x0 = VIEW_W + 20
        self.sliders = [Slider("Pitch", (x0, 50, PANEL_W - 40, 10), -args.limit, args.limit),
                        Slider("Yaw  ", (x0, 110, PANEL_W - 40, 10), -args.limit, args.limit)]
        self.stiff = Slider("Stiffness E", (x0, 170, PANEL_W - 40, 10), args.e_min, args.e_max,
                            unit="MPa", step=0.1, fmt="{:5.1f}")
        if args.youngs_mpa is not None:
            self.sim.set_E(args.youngs_mpa)
        self.stiff.value = round(self.sim.E, 1)
        self.btn_center = pygame.Rect(x0, 200, 90, 26)
        self.btn_refresh = pygame.Rect(x0 + 100, 200, 90, 26)
        self.port_y0 = 250
        self.echo = args.no_serial
        self.config, self.discs = args.config, args.discs
        self.bridge = None  # open ServoBridge, or None = sim only
        self.selected = NO_BOARD
        self.opening = None  # (thread, port) while a port is being opened
        self.last_sent = None
        self.msg, self.msg_warn = "sim only: nothing is sent", False
        self.closing = False
        self.fps = 0.0
        self.refresh_ports()
        if args.port and not args.no_serial:
            self.select(args.port)

    # ---- serial -------------------------------------------------------
    def refresh_ports(self):
        try:
            from serial.tools import list_ports
            found = [(p.device, p.description) for p in list_ports.comports()]
        except Exception as e:  # pyserial missing / OS error: keep sim usable
            found, self.msg, self.msg_warn = [], f"port scan failed: {e}", True
        self.ports = [(NO_BOARD, "")] + found
        if self.selected not in [p for p, _ in self.ports]:
            self.ports.append((self.selected, "(not currently listed)"))

    def _close_bridge(self):
        if self.bridge is not None:
            try:
                self.bridge.close()
            except Exception:
                pass
        self.bridge, self.last_sent = None, None

    def select(self, port):
        if self.opening:
            return  # one open at a time
        self._close_bridge()
        self.selected = port
        if port == NO_BOARD:
            self.msg, self.msg_warn = "sim only: nothing is sent", False
            return
        self.msg, self.msg_warn = f"opening {port} (2 s reset wait)...", False
        result = {}

        def worker():
            try:
                result["bridge"] = sb.ServoBridge(port, config=self.config, discs=self.discs)
            except Exception as e:
                result["error"] = e

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        self.opening = (t, port, result)

    def _poll_opening(self):
        if not self.opening or self.opening[0].is_alive():
            return
        _, port, result = self.opening
        self.opening = None
        if "bridge" in result and self.selected == port:
            self.bridge, self.last_sent = result["bridge"], None  # resend current state
            self.msg, self.msg_warn = f"connected {port}: sending", False
        else:
            if "bridge" in result:
                result["bridge"].close()
            self._fail(f"open {port} failed: {result.get('error')}")

    def _fail(self, text):
        self._close_bridge()
        self.selected = NO_BOARD
        self.msg, self.msg_warn = f"{text} -> sim only", True

    def _send_if_changed(self):
        line = self.ctrl.line
        if line == self.last_sent:
            return
        if self.bridge is not None:
            try:
                theta, phi = self.ctrl.bend()
                self.bridge.send(theta, phi)
            except Exception as e:  # unplugged mid-session, write timeout, ...
                self._fail(f"write failed: {e}")
                return
        elif self.echo:
            print(repr(line) + ("  CLAMPED" if self.ctrl.clamped else ""), flush=True)
        self.last_sent = line

    # ---- input --------------------------------------------------------
    def handle(self, e):
        if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
            self.close()
            return
        if self.closing:
            return
        for s in self.sliders + [self.stiff]:
            s.handle(e)
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            if self.btn_center.collidepoint(e.pos):
                self.center()
            elif self.btn_refresh.collidepoint(e.pos):
                self.refresh_ports()
            for i, (port, _) in enumerate(self.ports):
                if self._port_rect(i).collidepoint(e.pos) and port != self.selected:
                    self.select(port)
        if e.type == pygame.KEYDOWN:
            step = 5 if e.mod & pygame.KMOD_SHIFT else 1
            keys = {pygame.K_UP: (0, step), pygame.K_DOWN: (0, -step),
                    pygame.K_RIGHT: (1, step), pygame.K_LEFT: (1, -step)}
            if e.key in keys:
                i, dv = keys[e.key]
                s = self.sliders[i]
                s.value = max(s.lo, min(s.hi, s.value + dv))
            elif e.key == pygame.K_SPACE:
                self.center()
            elif e.key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
                dv = 0.1 if e.key == pygame.K_RIGHTBRACKET else -0.1
                self.stiff.value = round(max(self.stiff.lo, min(self.stiff.hi, self.stiff.value + dv)), 1)

    def center(self):
        for s in self.sliders:
            s.value = 0.0

    def close(self):
        """Ramp back to straight at the slew rate, then close the port."""
        self.closing = True
        self.center()

    # ---- frame --------------------------------------------------------
    def _port_rect(self, i):
        return pygame.Rect(VIEW_W + 20, self.port_y0 + 22 + i * 24, PANEL_W - 40, 22)

    def frame(self, dt):
        """One tick. Returns False once the close ramp has finished."""
        self._poll_opening()
        if abs(self.stiff.value - self.sim.E) > 1e-9:
            self.sim.set_E(self.stiff.value)
        self.ctrl.set_target(self.sliders[0].value, self.sliders[1].value)
        self.ctrl.tick(dt)
        self._send_if_changed()
        self.sim.apply(self.ctrl.angles)
        self.sim.step(dt)
        rgb = self.sim.render()
        if self.closing and self.ctrl.settled() and not self.opening:
            self._close_bridge()
            return False
        self.draw(rgb)
        return True

    def draw(self, rgb):
        scr = self.screen
        scr.fill(BG)
        scr.blit(pygame.image.frombuffer(rgb, (VIEW_W, VIEW_H), "RGB"), (0, 0))
        for s in self.sliders + [self.stiff]:
            s.draw(scr, self.font)
        for r, t in ((self.btn_center, "Center"), (self.btn_refresh, "Refresh")):
            pygame.draw.rect(scr, (60, 62, 70), r, border_radius=4)
            scr.blit(self.font.render(t, True, FG), (r.x + 12, r.y + 5))
        scr.blit(self.font.render("Serial port", True, FG), (VIEW_W + 20, self.port_y0))
        for i, (port, desc) in enumerate(self.ports):
            r = self._port_rect(i)
            sel = port == self.selected
            pygame.draw.rect(scr, (45, 80, 120) if sel else (40, 42, 48), r, border_radius=3)
            label = ("* " if sel else "  ") + port + (f"  {desc}" if desc and desc != "n/a" else "")
            scr.blit(self.small.render(label[:52], True, FG), (r.x + 6, r.y + 4))

        p, y = self.ctrl.current
        theta, phi = self.ctrl.bend()
        tip_t, tip_p = self.sim.tip_bend()
        a = self.ctrl.angles
        sent = self.bridge.sent if self.bridge else 0
        lines = [
            (f"cmd pitch {p:+6.1f} yaw {y:+6.1f} -> bend {theta:5.1f} deg @ {phi:+6.1f} | "
             f"sim tip {tip_t:5.1f} deg @ " + (f"{tip_p:+6.1f}" if tip_t > 0.5 else "   -- ") +
             f" (E {self.sim.E:g} MPa)", FG),
            ("servos " + "  ".join(f"A{i} {round(v):3d}" for i, v in enumerate(a)) +
             f"   sent {sent}" + ("   CLAMPED" if self.ctrl.clamped else "") +
             f"   {self.fps:4.1f} fps", WARN if self.ctrl.clamped else FG),
            (("closing: ramping to straight...  " if self.closing else "") + self.msg,
             WARN if self.msg_warn else DIM),
        ]
        for i, (t, c) in enumerate(lines):
            scr.blit(self.font.render(t, True, c), (10, VIEW_H + 6 + i * 20))
        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()
        dt, t_fps, n = 1.0 / TICK_HZ, time.perf_counter(), 0
        try:
            while True:
                for e in pygame.event.get():
                    self.handle(e)
                if not self.frame(dt):
                    break
                dt = clock.tick(TICK_HZ) / 1000.0
                n += 1
                now = time.perf_counter()
                if now - t_fps >= 1.0:
                    self.fps, t_fps, n = n / (now - t_fps), now, 0
        finally:
            self._close_bridge()
            self.sim.close()
            pygame.quit()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", help="pre-select this serial port at startup")
    p.add_argument("--no-serial", action="store_true",
                   help="pre-select 'No board' and echo each line to stdout")
    p.add_argument("--config", default=str(sb.DEFAULT_CONFIG))
    p.add_argument("--discs", type=int, default=sb.DEFAULT_DISCS)
    p.add_argument("--rate", type=float, default=90.0, help="max command slew (deg/s)")
    p.add_argument("--limit", type=float, default=60.0, help="slider range +/- deg")
    p.add_argument("--youngs-mpa", type=float, help="starting backbone E (default: from config)")
    p.add_argument("--e-min", type=float, default=0.5, help="stiffness slider min (MPa)")
    p.add_argument("--e-max", type=float, default=10.0, help="stiffness slider max (MPa)")
    App(p.parse_args()).run()


if __name__ == "__main__":
    main()
