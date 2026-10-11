# girlshackhard

Girlshackhard project: a tendon-driven continuum robot (TDCR) — a soft silicone
tube backbone threaded through 3D-printed discs, bent in pitch and yaw by two
servo-driven tendon pairs. This repo has the MuJoCo simulation used to size the
design, a live pitch/yaw control UI (with a predicted shape, works without a
board), the serial bridge to the servos, and the firmware.

## Current design

| Part | Value |
|---|---|
| Segments | 1 |
| Discs | 8 × PLA, Ø40 mm, 3 mm thick, Ø8.5 mm bore, 15% infill (≈2.9 g each) |
| Disc pitch | 20 mm centre-to-centre (17 mm face-to-face gap) → 160 mm segment |
| Backbone | silicone tube, Ø8.5 mm OD / Ø6.5 mm ID |
| Tendons | 4, through Ø1.75 mm holes at 17 mm radius, 0/90/180/270° |
| Actuation | 2 × SG90 servos, one per agonist–antagonist pair: A0 = pitch (0/180°), A1 = yaw (90/270°) |
| Spool | Ø50 mm (r = 25 mm) |
| Servo range | 30–150° (rest = 90°, set by hand at pretension when threading) |

In simulation (hanging, gravity, real disc mass) a 60° pitch/yaw command reaches
≈59° at the tip, settles in ≈3–4 s, and the worst servo load is ≈13% of SG90
stall (`scripts/one_seg_check.py`). Gravity makes the bend collect toward the tip
(≈2° per disc gap near the mount, ≈15° near the tip) instead of an even arc;
without gravity it is an even 7.5° per gap.

## Setup

Needs [uv](https://docs.astral.sh/uv/). The simulator
([opencr-mujoco](https://github.com/ContinuumRoboticsLab/opencr-mujoco)) is a
git submodule, so clone recursively:

```bash
git clone --recursive https://github.com/lok3cha4/girlshackhard
cd girlshackhard
uv sync          # Python 3.12, mujoco 3.14.0, pygame, pyserial
```

(Already cloned without `--recursive`? Run `git submodule update --init`.)

## Run

<img width="914" height="546" alt="image" src="https://github.com/user-attachments/assets/b331a5fc-f05e-4aea-ac01-7c3a65bdaa4f" />


**Live control UI** (pygame window: pitch/yaw sliders, live MuJoCo-predicted
shape, serial port picker):

```bash
uv run python scripts/bend_ui.py                   # starts in "No board / sim only"
uv run python scripts/bend_ui.py --port /dev/cu.X  # pre-select a board
```

- Starts in **sim only**: nothing is opened or sent. Pick a port from the list
  to connect (opens in the background, 2 s board-reset wait); pick "No board"
  to disconnect. Press **Refresh** after plugging a board in.
- Commands are slew-limited (90°/s, `--rate`) and a line is only sent when a
  servo angle changes. Closing the window ramps back to straight first.
- **Stiffness slider:** sets the tube's Young's modulus live (0.5–10 MPa) to see
  how a stiffer or softer backbone changes the predicted shape (`[` / `]` = ∓0.1 MPa).
- Keys: arrows nudge 1° (Shift: 5°), Space centers the bend, Esc closes.

**2-segment sim** (sim only, no serial): 2 × 8 discs, 2 × 160 mm, four
pitch/yaw sliders plus the stiffness slider; G toggles gravity to compare with
the ideal constant-curvature shape.

```bash
uv run python scripts/bend_ui_2seg.py
```

**Z insertion + obstacle course** (sim only): `bend_ui_2seg_z.py` adds a Z slider
(advance / retract); `bend_ui_2seg_course.py` adds a winding vessel-like tube the
robot has to be pushed and steered through (discs collide with the walls; contact
force, progress, target distance and a timer are shown).

**Any robot size** (sim only): `bend_ui_nseg.py` adds sliders for the number of
segments (1–4) and segment length (40–200 mm, 20 mm disc pitch); changing either
rebuilds the model in about 0.4 s. Use `--no-course` for free space.

```bash
uv run python scripts/bend_ui_2seg_course.py
uv run python scripts/bend_ui_nseg.py --segments 3 --seg-len 120
```

**Single command from the terminal:**

```bash
uv run python scripts/serial_bridge.py --theta 60 --no-serial        # print the line
uv run python scripts/serial_bridge.py --theta 60 --phi 90 --port /dev/cu.X
```

`--theta` is the bend (°), `--phi` its direction (0 = pitch toward A0, 90 = yaw
toward A1). For a control loop use `ServoBridge` (opens the port once) instead
of calling the script per command.

**Simulation checks:**

```bash
uv run python scripts/one_seg_check.py   # 60° / diagonal bend, shape, settling, SG90 load
```

## Firmware

`FIRMWARE/firmware.cpp` (Arduino-style, Axiometa board, ST7735 display).

- Serial protocol, 115200 baud: `A0:deg,A1:deg\n` (integer degrees). The board
  replies `OK a,b` to each line; the bridge discards the replies.
- Clamps every command to `SAFE_MIN_DEG`/`SAFE_MAX_DEG` = 30/150. Keep these in
  sync with `SAFE_MIN`/`SAFE_MAX` in `scripts/serial_bridge.py`.
- **Before flashing:** set `PITCH_SERVO_PIN` and `YAW_SERVO_PIN` (the build
  stops with an `#error` until both are defined).

## First board test

1. Fill in the two servo pins and flash the firmware.
2. With tendons slack or lightly tensioned, send small moves and check each
   spool winds the right way:
   `--theta 5` (A0 should pull the 0° tendon) and `--theta 5 --phi 90` (A1, 90°).
   If one is reversed, flip its entry in `SERVO_SIGN` in `serial_bridge.py`.
3. Then use `bend_ui.py`.

## Repo layout

```
scripts/
  bend_ui.py          live pygame UI (sliders, stiffness, predicted shape, port picker)
  bend_ui_2seg.py     sim-only 2-segment version (4 bend sliders, stiffness, gravity toggle)
  bend_ui_2seg_z.py   + Z (insertion) slider
  bend_ui_2seg_course.py  + anatomy-like obstacle course with contact feedback
  bend_ui_nseg.py     course sim with segment-count and segment-length sliders
  serial_bridge.py    bend -> 2 servo angles, serial protocol, ServoBridge
                      (R_SPOOL, SAFE_MIN/MAX, SERVO_MAP live here — single copy)
  one_seg_check.py    current design: bend, shape, settling, SG90 load
  sim_disc_design.py  builds the MuJoCo model: disc mass + disc-hole tendon routing
  bend60_check.py     2-segment 60° check (earlier design studies)
  spool_sizing.py     2-segment spool sizing (earlier design studies)
  sweep_2seg.py       2-segment disc/length sweep + shared sim helpers
  render_bend.py      early single-segment render helper
configs/generation/
  one_segment_8disc.json      current hardware layout
  two_segment_8disc.json      2-segment sim layout (mounted at 0.5 m to clear the floor)
  single_segment_4tendon.json base material/mount config (tube, E = 2 MPa)
FIRMWARE/firmware.cpp
third_party/opencr-mujoco     simulator (submodule, pinned)
```

Scripts write videos/CSVs/scene XML to `outputs/` and `assets/` (git-ignored,
regenerable). `third_party/` is not modified.

## Caveats / placeholders

- **Length:** the sim uses a 160 mm segment (8 discs at 20 mm pitch, first disc
  one pitch from the mount, tip disc at the tube end). Free tube between the
  mount and the first disc, or past the last disc, isn't modelled yet.
- **Tube stiffness** E = 2 MPa and **ID** 6.5 mm are estimates; a sideways sag
  test of the real tube with discs would calibrate E.
- **Disc mass** is an estimate from geometry and infill (±50%); weigh one.
- **Friction** between tendon and disc holes isn't simulated; loads above
  include a guessed capstan factor (μ = 0.25).
- **Servo direction** (`SERVO_SIGN`) is unverified on hardware.
- **Resolution:** 1 servo degree ≈ 0.44 mm of tendon ≈ 1.5° of bend with the
  Ø50 mm spool.
