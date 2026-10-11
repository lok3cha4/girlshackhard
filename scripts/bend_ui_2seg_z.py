"""2-segment sim (bend_ui_2seg.py) plus a Z slider that moves the whole robot
up / down -- the first step toward navigating it through a body (advance /
retract along the insertion axis while steering both segments).

Everything from bend_ui_2seg.py is reused (four pitch/yaw sliders, live
stiffness, virtual 4-servo mapping, gravity toggle). The robot's mount is made
a MuJoCo mocap body (via the spec_edit hook, before compiling), so its height
is set kinematically each tick and the soft robot follows physically. The Z
command is slew-limited (--z-rate mm/s) like the bend commands.

Z is the mount's height relative to its start (+ = up). The robot hangs
straight down, so this is also the insertion axis: Z down = advance.

    uv run python scripts/bend_ui_2seg_z.py
Keys: as bend_ui_2seg.py, plus R / F = up / down 1 mm (shift: 10 mm),
H = home Z.
"""

import argparse

import numpy as np
import pygame

import bend_ui as ui
import bend_ui_2seg as u2
import sweep_2seg as sw


def _make_mount_mocap(spec):
    spec.body("tdcr_mount").mocap = True


class SimViewZ(u2.SimView2):
    """2-segment model whose mount (tdcr_mount) is a mocap body moved in Z."""

    def __init__(self, config=u2.CONFIG):
        super().__init__(config, spec_edit=_make_mount_mocap)
        self.mocap_id = self.m.body_mocapid[self.m.body("tdcr_mount").id]
        self.base0 = self.d.mocap_pos[self.mocap_id].copy()

    def set_z(self, z_mm):
        self.d.mocap_pos[self.mocap_id] = self.base0 + np.array([0.0, 0.0, z_mm / 1000.0])

    def tip_world(self):
        return self.d.xpos[self.m.body("EE_pos").id].copy()

    def lowest_point(self):
        """Lowest robot point above the floor (m): links and discs."""
        ids = [i for i in range(self.m.nbody) if self.m.body(i).name.startswith(("link_", "disc_"))]
        return float(min(self.d.xpos[i][2] for i in ids))


class AppZ(u2.App):
    HINT_Y = 460
    PANEL_EXTRA_H = 20  # room for the extra base-Z status line

    def make_sim(self, args):
        return SimViewZ(args.config)

    def __init__(self, args):
        super().__init__(args)
        x0, w = u2.VIEW_W + 20, ui.PANEL_W - 40
        self.zs = ui.Slider("Z (up/down)", (x0, 395, w, 10), -args.z_down, args.z_up,
                            unit="mm", step=1, fmt="{:+5.0f}")
        self.btn_center.y = 420
        self.z_cur, self.z_rate = 0.0, args.z_rate
        pygame.display.set_caption("2-segment TDCR - pitch / yaw / Z - live sim (no board)")

    def widgets(self):
        return super().widgets() + [self.zs]

    def handle(self, e):
        super().handle(e)
        if self.closing or e.type != pygame.KEYDOWN:
            return
        step = 10 if e.mod & pygame.KMOD_SHIFT else 1
        if e.key in (pygame.K_r, pygame.K_f):
            dz = step if e.key == pygame.K_r else -step
            self.zs.value = max(self.zs.lo, min(self.zs.hi, self.zs.value + dz))
        elif e.key == pygame.K_h:
            self.zs.value = 0.0

    def frame(self, dt):
        step = self.z_rate * dt
        self.z_cur += max(-step, min(step, self.zs.value - self.z_cur))
        self.sim.set_z(self.z_cur)
        return super().frame(dt)

    def hints(self):
        h = super().hints()
        return h[:3] + ["R / F: up / down (shift 10 mm)  H: home Z"] + h[3:]

    def status_lines(self):
        lines = super().status_lines()
        tip = self.sim.tip_world() * 1000
        low = self.sim.lowest_point() * 1000
        z_line = (f"base Z {self.z_cur:+6.1f} mm (cmd {self.zs.value:+.0f})   tip world "
                  f"({tip[0]:+6.1f}, {tip[1]:+6.1f}, {tip[2]:6.1f}) mm   lowest point {low:5.0f} mm above floor",
                  ui.WARN if low < 5 else ui.FG)
        return lines[:-1] + [z_line, lines[-1]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(u2.CONFIG))
    p.add_argument("--rate", type=float, default=90.0, help="max bend slew (deg/s)")
    p.add_argument("--limit", type=float, default=60.0, help="bend slider range +/- deg")
    p.add_argument("--youngs-mpa", type=float, help="starting backbone E (default: from config)")
    p.add_argument("--e-min", type=float, default=0.5, help="stiffness slider min (MPa)")
    p.add_argument("--e-max", type=float, default=10.0, help="stiffness slider max (MPa)")
    p.add_argument("--z-up", type=float, default=150.0, help="Z slider max (mm up)")
    p.add_argument("--z-down", type=float, default=150.0, help="Z slider max (mm down)")
    p.add_argument("--z-rate", type=float, default=50.0, help="max Z slew (mm/s)")
    AppZ(p.parse_args()).run()


if __name__ == "__main__":
    main()
