"""
dobotArm.py — Low-level control library for the Dobot Magician robotic arm.

Wraps the Dobot DLL (lib.DobotDllType) into easy-to-use Python functions:
  - connect, home, and initialize the robot
  - move to XYZ coordinates (blocking, with safety wait)
  - move by joint angles
  - rotate the end effector (wrist rotation)
  - open/close the gripper
  - set movement speed
  - stop the pneumatic pump

All movement functions default to isQueued=0 (immediate/synchronous mode),
which means the DLL call blocks until the physical motion finishes on this
robot's firmware. A safety sleep ensures we don't race ahead.
"""

import lib.DobotDllType as dType
import math
import time

# ── Connection status strings for display / debugging ──
# When ConnectDobot returns a status code, we map it to a human-readable name.
CON_STR = {
    dType.DobotConnect.DobotConnect_NoError:  "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"
}

# ── Load the Dobot DLL ──
# This MUST happen before any other Dobot calls. The 'api' handle is passed
# to every subsequent function so the DLL knows which robot instance to talk to.
api = dType.load()

# ── Home position (robot coordinates in mm) ──
# Where the robot goes after homing and between pick cycles.
# (200, 100, 50) is to the left of the robot's x-axis, slightly above the XY
# plane — a safe spot that keeps the arm out of the camera's view.
home_pos = [200, 100, 50]


def initialize_robot(api):
    """Connect to the Dobot over serial, home it, and get ready for movement.

    Steps:
      1. Search for the Dobot on available COM ports.
      2. Connect at 115200 baud on COM6 (hardcoded — edit if your robot is on a different port).
      3. Stop any leftover queued commands and clear the command queue.
      4. Set speed/acceleration to 50% of max (safe default).
      5. Set the home position coordinates.
      6. Enqueue and execute the homing command — the robot beeps when done.
      7. Wait (polling the queue index) until homing finishes.
    """
    com_port = dType.SearchDobot(api)[0]
    if "COM" not in com_port:
        print("Error: The robot either isn't on or isn't responding. Exiting now")
        exit()

    state = dType.ConnectDobot(api, "COM6", 115200)[0]
    if state != dType.DobotConnect.DobotConnect_NoError:
        print("Failed to connect to Dobot!")
        exit()

    # Clear any stale commands from a previous session — avoids dangerous
    # unexpected movement when we start executing the queue.
    dType.SetQueuedCmdStopExec(api)
    dType.SetQueuedCmdClear(api)

    dType.SetPTPCommonParams(api, 50, 50, isQueued=1)

    # Configure and run homing: robot resets encoders, moves to home_pos,
    # then validates its position sensing.
    dType.SetHOMEParams(api, home_pos[0], home_pos[1], home_pos[2], 0, isQueued=1)
    cmdIndx = -1
    execCmd = dType.SetHOMECmd(api, temp=0, isQueued=1)[0]

    dType.SetQueuedCmdStartExec(api)

    # Wait until ALL queued motion actually finishes (not just starts).
    # GetQueuedCmdMotionFinish returns True only when the physical motion
    # is complete — unlike GetQueuedCmdCurrentIndex which advances when a
    # command starts executing.
    while not dType.GetQueuedCmdMotionFinish(api)[0]:
        dType.dSleep(25)


def move_to_xyz(api, x, y, z, rHead=0, wait=True):
    """Move the robot arm to (x, y, z) in mm using joint-interpolated motion.

    Uses isQueued=0 (immediate mode). On this Dobot firmware, the DLL call
    appears to be synchronous — it blocks until motion is physically complete.

    The 3-second sleep is a safety buffer to guarantee the move finished.
    If wait=False, the command is fired and the function returns immediately;
    the caller is responsible for timing the wait.

    Returns 0 on success (no error reporting for now).
    """
    if not all(math.isfinite(v) for v in (x, y, z, rHead)):
        print(f"Refusing invalid Dobot move: {(x, y, z, rHead)}")
        return -1
    try:
        dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, x, y, z, rHead, isQueued=0)
    except Exception as e:
        print(f"Dobot move failed: {e}")
        return -1
    if wait:
        dType.dSleep(3000)
    return 0


def move_joint_angles(api, J1, J2, J3, J4=0):
    """Move the robot by specifying each joint angle directly (degrees).

    Uses PTPMOVJANGLEMode. J4 (wrist rotation) defaults to 0 since most
    pick-place tasks don't need it (the gripper orientation is handled
    separately by rotate_end_effector).

    NOTE: This still uses the old isQueued=1 + polling pattern. It is not
    used by collaborative_demo.py and may need updating for consistency.
    """
    cmdIndx = -1
    execCmd = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJANGLEMode, J1, J2, J3, J4, isQueued=0)[0]
    while execCmd > dType.GetQueuedCmdCurrentIndex(api)[0]:
        dType.dSleep(25)


def move_to_home(api):
    """Move robot to the stored home position using a normal PTP move.

    This does NOT re-run the homing/sensor-init procedure — it simply does
    a coordinated move to home_pos. Faster and safer than SetHOMECmd during
    normal operation (e.g. when shutting down).
    """
    move_to_xyz(api, home_pos[0], home_pos[1], home_pos[2])


def rotate_end_effector(api, angle, wait=True):
    """Rotate the wrist/gripper to the given angle (-90 to +90 degrees).

    Reads the current pose, then issues a linear move that keeps the
    TCP at the same XYZ but changes the wrist rotation (rHead).

    Uses isQueued=0 + 3s sleep, matching the pattern in move_to_xyz.
    Returns 0 on success, -1 if angle is out of range.
    """
    if 90 >= angle >= -90:
        try:
            pose = dType.GetPose(api)
            dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode,
                            pose[0], pose[1], pose[2], angle, isQueued=0)
        except Exception as e:
            print(f"Dobot wrist rotation failed: {e}")
            return -1
        if wait:
            dType.dSleep(3000)
        return 0
    return -1


def open_gripper(api):
    """Open the gripper (release whatever it's holding).

    The Dobot API provides no feedback for gripper state, so we just
    send the command and wait 500ms for it to physically open.
    """
    dType.SetEndEffectorGripper(api, 1, 0, 0)[0]
    dType.dSleep(500)


def close_gripper(api):
    """Close the gripper (grip whatever is between the fingers).

    Same pattern as open_gripper — send command, wait 500ms.
    """
    dType.SetEndEffectorGripper(api, 1, 1, 0)[0]
    dType.dSleep(500)


def set_speed(api, velocity_pct, acceleration_pct=None):
    """Set robot movement speed as a percentage of max (1–100).

    If acceleration_pct is not given, it matches velocity_pct.
    Both are clamped to [1, 100] to avoid extreme values.
    This affects all subsequent PTP moves.
    """
    if acceleration_pct is None:
        acceleration_pct = velocity_pct
    velocity_pct = max(1, min(100, velocity_pct))
    acceleration_pct = max(1, min(100, acceleration_pct))
    dType.SetPTPCommonParams(api, velocity_pct, acceleration_pct, isQueued=0)


def stop_pump(api):
    """Turn off the pneumatic suction cup (used to release blocks).

    Despite the function name mentioning "suction cup", on this setup it
    controls the pneumatic pump. We just send the off command and wait
    50ms — the gripper's open_gripper/close_gripper are the primary
    end-effector controls.
    """
    dType.SetEndEffectorSuctionCup(api, 1, 0, 0)[0]
    dType.dSleep(50)


def start_pump(api):
    # Start the suction pump (enable = 1, suction = 1)
    # This is a minimal wrapper so code can use either gripper or suction depending on hardware.
    dType.SetEndEffectorSuctionCup(api, 1, 1, 0)[0]
    # small delay to allow pump to spool up
    dType.dSleep(200)

