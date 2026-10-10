"""Serial bridge: single-segment constant-curvature bend -> 2 servo angles.

Protocol (matches FIRMWARE/firmware.cpp): "A0:deg,A1:deg\\n" at 115200 baud.

Hardware: 1 segment, 2 servos, one per agonist-antagonist tendon pair
(A0 = pitch pair, A1 = yaw pair). Each spool (R_SPOOL) winds the pair's
primary tendon in and pays the opposite one out. No sensing: rest = 90 deg by
convention (hand-calibrated at pretension when threading), so
angle = 90 + delta_servo, clamped to the firmware's SAFE_MIN/SAFE_MAX.

Tendon geometry comes from the generation config (no sim readback): tendon n
sits at angle seg_offsets[0] + n * 360/n_tendons, radius distance_to_backbone.
Order verified against the compiled generator model:
    seg_0_tendon_0..3 at 0 / 90 / 180 / 270 deg
so the pairs are (0, 2) -> A0 (pitch) and (1, 3) -> A1 (yaw).

    uv run python scripts/serial_bridge.py --theta 60 --no-serial
    uv run python scripts/serial_bridge.py --theta 60 --port /dev/cu.usbmodemXXXX

Live control loop: use ServoBridge (port opened once), not main() per tick.
"""

import argparse
import json
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/generation/one_segment_8disc.json"  # 8 discs, 80 mm
DEFAULT_DISCS = 8

BAUD = 115200
R_SPOOL = 0.025  # m (50 mm diameter spool) -- the only copy; everything reads this
REST_DEG = 90.0
SAFE_MIN, SAFE_MAX = 30.0, 150.0  # must match firmware SAFE_MIN_DEG / SAFE_MAX_DEG

# Ax -> (primary tendon, antagonist tendon) of the single segment. +delta winds
# the primary in (shortens it) and pays the antagonist out.
SERVO_MAP = [
    (0, 2),  # A0: pitch pair, holes at  0 / 180 deg
    (1, 3),  # A1: yaw pair,   holes at 90 / 270 deg
]
# Physical spool winding direction per servo (+1: increasing angle shortens
# the primary tendon). Unverified on hardware; flip an entry if a servo pulls
# the wrong way.
SERVO_SIGN = [+1, +1]


def load_layout(config_path):
    c = json.load(open(config_path))
    segs = []
    for s in range(c["num_segments"]):
        a = c["actuation_details"]["segments"][s]
        n = a["number_of_tendons"]
        off = c["seg_offsets"][s] if s < len(c.get("seg_offsets", [])) else 0.0
        segs.append(dict(
            L=c["segment_lengths"][str(s + 1)],
            d=a["distance_to_backbone"],
            angles=[off + i * 2 * math.pi / n for i in range(n)],
        ))
    return segs


def tendon_dl(segs, bends, discs):
    """Length change (m, + = longer) of every tendon for per-segment CC bends.

    bends: [(theta_rad, phi_rad)] per segment, phi = bend direction in the
    robot base frame (same frame as the tendon angles). Tendons run as
    straight chords between `discs` discs per segment. General in segment
    count (the sim/sizing scripts use it for 2-segment layouts too).
    """
    dl = {}
    for s, seg in enumerate(segs):
        for n, psi in enumerate(seg["angles"]):
            total = 0.0
            for k in range(s + 1):  # coupled routing through proximal segments
                th, phi = bends[k]
                if abs(th) < 1e-9:
                    continue
                Lk, R = segs[k]["L"], segs[k]["L"] / th
                x = seg["d"] * math.cos(psi - phi)
                total += discs * 2 * (R - x) * math.sin(th / discs / 2) - Lk
            dl[(s, n)] = total
    return dl


def servo_angles(segs, theta, phi, discs, r_spool=R_SPOOL):
    """Single-segment bend (radians) -> (clamped angles, raw angles, pair dl mm)."""
    if len(segs) != 1:
        raise ValueError(f"2-servo hardware drives exactly 1 segment; layout has {len(segs)}")
    dl = tendon_dl(segs, [(theta, phi)], discs)
    raw, pairs = [], []
    for (p, a), sign in zip(SERVO_MAP, SERVO_SIGN):
        pull = (dl[(0, a)] - dl[(0, p)]) / 2  # m wound in on the primary
        raw.append(REST_DEG + sign * math.degrees(pull / r_spool))
        pairs.append((round(dl[(0, p)] * 1000, 2), round(dl[(0, a)] * 1000, 2)))
    clamped = [min(SAFE_MAX, max(SAFE_MIN, v)) for v in raw]
    return clamped, raw, pairs


def format_line(angles):
    """Exactly len(SERVO_MAP) fields: "A0:deg,A1:deg\\n"."""
    if len(angles) != len(SERVO_MAP):
        raise ValueError(f"expected {len(SERVO_MAP)} angles, got {len(angles)}")
    return ",".join(f"A{i}:{round(v)}" for i, v in enumerate(angles)) + "\n"


class ServoBridge:
    """Persistent bridge for the live control loop.

    Opens the port once (plus the 2 s board-reset wait) and loads the tendon
    layout once; every send() then only computes angles and writes one line.

        with ServoBridge("/dev/cu.usbmodem1101") as bridge:
            while running:
                bridge.send(theta, phi)   # degrees

    no_serial=True prints each line instead of writing (no port opened).
    `port` may also be a pyserial URL, e.g. "loop://" for a loopback test.
    """

    def __init__(self, port=None, config=DEFAULT_CONFIG, discs=DEFAULT_DISCS, no_serial=False,
                 reset_wait=2.0, write_timeout=0.1):
        self.segs = load_layout(config)
        if len(self.segs) != 1:
            raise ValueError(f"{config}: 2-servo hardware needs a 1-segment layout")
        self.discs = discs
        self.no_serial = no_serial
        self.ser = None
        self.sent = 0
        self.clamped = 0  # sends where at least one servo hit SAFE_MIN/MAX
        self.last_reply = b""
        if no_serial:
            return
        if not port:
            raise ValueError("port is required unless no_serial=True")
        import serial

        self.ser = serial.serial_for_url(port, BAUD, timeout=0, write_timeout=write_timeout)
        time.sleep(reset_wait)  # most Arduino-style boards reset when the port opens
        self.ser.reset_input_buffer()

    def send(self, theta, phi):
        """Write one command (degrees). Returns (line, clamped: bool)."""
        angles, raw, _ = servo_angles(self.segs, math.radians(theta), math.radians(phi), self.discs)
        line = format_line(angles)
        clamped = any(abs(r - c) > 1e-9 for r, c in zip(raw, angles))
        self.clamped += clamped
        self.sent += 1
        if self.ser is None:
            print(repr(line) + ("  CLAMPED" if clamped else ""))
        else:
            self.ser.write(line.encode("ascii"))
            # firmware answers every line with "OK a,b"; drain so replies
            # never back up (keep the newest for debugging)
            if self.ser.in_waiting:
                self.last_reply = self.ser.read(self.ser.in_waiting)
        return line, clamped

    def close(self):
        if self.ser is not None:
            self.ser.close()
            self.ser = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--theta", type=float, default=0.0, help="bend (deg)")
    p.add_argument("--phi", type=float, default=0.0,
                   help="bend direction (deg): 0 = pitch toward A0's primary hole, 90 = yaw toward A1's")
    p.add_argument("--discs", type=int, default=DEFAULT_DISCS, help="discs in the segment (tendon chords)")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--port", help="serial port, e.g. /dev/cu.usbmodem1101")
    p.add_argument("--no-serial", action="store_true", help="print the line instead of writing it")
    args = p.parse_args()

    segs = load_layout(args.config)
    angles, raw, pairs = servo_angles(segs, math.radians(args.theta), math.radians(args.phi), args.discs)
    line = format_line(angles)

    for i, ((pr, an), r, c, pd) in enumerate(zip(SERVO_MAP, raw, angles, pairs)):
        flag = "  CLAMPED" if abs(r - c) > 1e-9 else ""
        print(f"A{i} t{pr}/t{an}: dl {pd[0]:+.2f}/{pd[1]:+.2f} mm -> {r:7.1f} deg (sent {round(c)}){flag}")
    if any(abs(r - c) > 1e-9 for r, c in zip(raw, angles)):
        span = math.radians(SAFE_MAX - REST_DEG) * R_SPOOL * 1000
        print(f"WARNING: clamped to [{SAFE_MIN:g}, {SAFE_MAX:g}]; +/-{SAFE_MAX - REST_DEG:g} deg "
              f"x r_spool {R_SPOOL * 1000:g} mm = only +/-{span:.2f} mm of tendon travel")

    if args.no_serial:
        print(repr(line))
        return
    if not args.port:
        p.error("--port is required unless --no-serial")
    import serial

    with serial.Serial(args.port, BAUD, timeout=1) as ser:
        time.sleep(2.0)  # most Arduino-style boards reset when the port opens
        ser.write(line.encode("ascii"))
        ser.flush()
    print(f"sent to {args.port}: {line!r}")


if __name__ == "__main__":
    main()
