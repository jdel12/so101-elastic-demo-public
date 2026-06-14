# Building the Robot

This guide walks you through sourcing parts, assembling the SO-101 arms,
mounting cameras, and getting everything physically ready before you touch
any software.

## What you are building

The SO-101 is a 6-DOF (six degrees of freedom) robot arm designed by
HuggingFace for the LeRobot project. It uses Feetech STS3215 servos and
3D-printed structural parts. You need two arms:

- **Follower arm** — the one the AI model controls. All six servos use the
  same 1/345 gear ratio.
- **Leader arm** — the one you physically move during training data
  collection. Uses three different gear ratios (1/345, 1/191, 1/147) so it
  can hold its own weight while remaining easy to move by hand.

The leader arm is only needed for recording training data. If you already
have a trained model and a dataset, you can skip it — but we recommend
building both. You will want to record your own training data for your
specific setup.

## Sourcing parts

### Option 1: Buy a kit

Several vendors sell pre-assembled or kit versions of the SO-101:

- **Seeed Studio** — ships globally, includes 3D-printed parts
- **WowRobo** — includes servos and electronics
- **PartaBot** — EU-based supplier
- **Autodiscovery** — includes cameras

Kit prices range from $300-400 including shipping. This is the fastest path
if you don't have access to a 3D printer and don't want to source parts
individually.

### Option 2: Source parts yourself

| Component | Qty | Approx. cost | Notes |
|-----------|-----|-------------|-------|
| Feetech STS3215 servo, 7.4V, 1/345 ratio | 7 | ~$97 | 6 for follower + 1 for leader shoulder |
| Feetech STS3215 servo, 7.4V, 1/191 ratio | 2 | ~$28 | Leader shoulder pan + elbow |
| Feetech STS3215 servo, 7.4V, 1/147 ratio | 3 | ~$42 | Leader wrist roll, pitch, gripper |
| Feetech motor control board | 2 | ~$21 | One per arm |
| USB-C cables | 2 | ~$7 | Connect control boards to computer |
| 5V power supply (for 7.4V motors) | 2 | ~$20 | **Critical: do NOT use 12V — it will burn the motors** |
| Table clamps (4-pack) | 1 | ~$9 | Mount arms to table edge |
| Precision screwdriver set | 1 | ~$6 | M2 and M3 screws |

Total for both arms, sourced individually: **~$230**

Order servos from the Feetech official store or authorized resellers. The
STS3215 is commonly available on AliExpress, Amazon, and robotics suppliers.
Lead times from China are typically 2-3 weeks.

### 3D-printed parts

The structural parts are 3D-printed. STL files are provided in the LeRobot
repository.

**If you have a 3D printer:**
- PLA+ filament (PLA works but PLA+ is more durable)
- 0.4mm nozzle, 0.2mm layer height, 15% infill
- Support material everywhere except slopes over 45 degrees
- Each arm prints as multiple parts — expect 8-12 hours per arm
- Standard desktop printers work fine: Bambu Lab, Prusa, Creality Ender 3

**If you don't have a 3D printer:**
- Online printing services (JLCPCB, Craftcloud, Xometry) will print and
  ship parts for $30-60 per arm
- Many public libraries and makerspaces have 3D printers available
- Some of the kit vendors above include printed parts
- University fab labs are often open to community members

## Assembly

The HuggingFace LeRobot project maintains the authoritative assembly guide:

> **[SO-101 Assembly Guide](https://github.com/huggingface/lerobot/blob/main/examples/robots/so101/README.md)**

Follow that guide for the step-by-step assembly process. What follows here
are the things the guide doesn't tell you, or things we learned the hard way.

### Before you start: configure the motors

**This must happen before assembly.** Once the servos are installed in the
arm, the connectors are not accessible for individual configuration.

Each servo needs a unique ID (1-6) and a baudrate of 1,000,000. Connect
each servo individually via USB and use the LeRobot setup tool:

```bash
pip install lerobot[feetech]
lerobot-setup-motors
```

The tool walks you through assigning IDs. Label each servo with its ID
number (a piece of tape works) before installing it.

**Pro tip:** Test each servo individually before assembly. Apply a small
position command and verify it moves smoothly. A dead servo buried inside a
fully assembled arm means disassembling the whole thing.

### Assembly pro tips

**Time estimate:** First arm takes 1.5-2 hours. Second arm under 1 hour
once you know the drill. Budget half a day including motor configuration
and calibration.

**Joint assembly pattern:** Every joint follows the same sequence: seat
the motor in its holder, fasten with M2x6mm screws, attach the motor horn
with M3x6mm screws, mount the next structural piece. Learn the pattern on
joint 1 and the rest go faster.

**Wire routing matters.** The daisy-chain cables between servos can get
pinched if you route them poorly. Use the wire guides printed into the
structural parts. Route cables so they don't interfere with joint rotation
— a cable that gets caught in a rotating joint will break or stall the motor.

**The gripper is the fiddliest part.** The follower arm gripper uses a
servo-driven mechanism that needs careful alignment. Take your time here.
If the gripper binds or doesn't close fully, the model will learn bad
gripper behavior and you'll see "gripper mode collapse" in your training
results.

**Arm mounting:** Clamp both arms to a sturdy table edge with at least 18
inches of clear workspace in front. The arms need to be firmly mounted —
any wobble shows up as noise in your training data and makes the model's
job harder.

## Calibration

After assembly, calibrate both arms using LeRobot:

```bash
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyUSB0 \
    --robot.id=my_follower
```

Repeat for the leader arm. The calibration process asks you to move each
joint to specific positions to establish the range of motion. Calibration
data is saved in `~/.cache/calibration/so101/` and identified by the `id`
you choose.

**Use the same ID consistently.** The calibration ID must match across
teleoperation, recording, and evaluation. Pick a simple name and stick
with it.

**Verify with teleoperation.** After calibrating both arms, run a quick
teleoperation test:

```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyUSB0 \
    --robot.id=my_follower \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyUSB1 \
    --teleop.id=my_leader
```

Move the leader arm through its full range. The follower should mirror
smoothly. If any joint stutters, stalls, or moves in the wrong direction,
check the motor ID assignments and re-calibrate.

## Cameras

You need USB webcams for visual input to the VLA model. Our setup uses
three cameras:

| Camera | Mount position | Purpose |
|--------|---------------|---------|
| cam0 (wrist) | Near end effector or on the arm | Primary input to VLA model |
| cam1 (overhead) | Fixed above workspace | Evaluation, CLIP scoring |
| cam2 (third) | Side angle | Additional training context |

**What cameras to buy:** Any V4L2-compatible USB webcam at 640x480 or
higher. We use generic 1080p USB webcams ($15-25 each). Logitech C920/C922
are solid choices. You don't need anything expensive — the VLA model
downsamples frames to 224x224 anyway.

**The critical constraint is USB bandwidth.** Two cameras on the same USB
root hub will cause frame drops because USB 2.0 bandwidth is shared per
controller. Each camera needs its own USB controller.

Verify camera separation:

```bash
lsusb -t
```

Each camera should appear on a different bus number. If they're on the same
bus, plug them into ports that route to different controllers. On a desktop
motherboard, front-panel USB and rear USB are usually on different
controllers. On a NUC or laptop, you may need a USB PCIe card.

**Camera mounting:** Mount cameras rigidly. A camera that shifts between
training and inference means the model sees a different perspective than
what it trained on. Use clamps, not tape.

## What you should have at the end of this stage

- [ ] Two assembled SO-101 arms (leader + follower), clamped to a table
- [ ] Both arms calibrated with LeRobot (`lerobot-calibrate`)
- [ ] Teleoperation verified (leader moves, follower mirrors)
- [ ] 1-3 USB cameras mounted and verified (`v4l2-ctl --list-devices`)
- [ ] Cameras on separate USB controllers (`lsusb -t`)
- [ ] All parts sourced and functional

Next: [02-compute.md](02-compute.md) — setting up your GPU server and edge
node.
