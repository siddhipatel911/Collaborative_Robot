"""mood_movement.py

Map simple mood names to small, safe Dobot movement sequences. Each
movement is executed synchronously but wrapped in a background thread
so the UI can trigger motions without blocking the web server.

Functions:
  perform_mood_movement(api, mood) -> threading.Thread
"""
import threading
import time
import dobotArm


def _safe_move(api, x, y, z, wait=True):
    try:
        return dobotArm.move_to_xyz(api, x, y, z, 0, wait)
    except Exception:
        return -1


def _spin(api, times=1, speed_delay=0.4):
    # small wrist rotations left/right to simulate a spin
    for _ in range(times):
        dobotArm.rotate_end_effector(api, 45)
        time.sleep(speed_delay)
        dobotArm.rotate_end_effector(api, -45)
        time.sleep(speed_delay)
    dobotArm.rotate_end_effector(api, 0)


def _nod(api):
    pose = dobotArm.api and dobotArm.api or None
    # Use move_to_xyz to dip the arm slightly and return
    try:
        p = dobotArm.api
    except Exception:
        p = None
    # Best-effort: move down a little and back up
    try:
        dobotArm.move_to_xyz(dobotArm.api, dobotArm.home_pos[0], dobotArm.home_pos[1], dobotArm.home_pos[2] - 15)
        time.sleep(0.6)
        dobotArm.move_to_xyz(dobotArm.api, dobotArm.home_pos[0], dobotArm.home_pos[1], dobotArm.home_pos[2])
    except Exception:
        pass


def _left_right(api, distance=30, repeats=2, delay=0.4):
    # move left and right around home_x (small XY nudges)
    try:
        home = dobotArm.home_pos
        for _ in range(repeats):
            _safe_move(api, home[0] - distance, home[1], home[2])
            time.sleep(delay)
            _safe_move(api, home[0] + distance, home[1], home[2])
            time.sleep(delay)
        _safe_move(api, home[0], home[1], home[2])
    except Exception:
        pass


def _fast_spin(api):
    _spin(api, times=4, speed_delay=0.12)


def perform_mood_movement(api, mood):
    """Trigger a movement pattern for the given mood in a background thread.

    Returns the Thread object so callers can join if they want to wait.
    """
    mood = (mood or "unknown").lower()

    def runner():
        try:
            if mood == 'happy':
                # gentle spin
                _spin(api, times=2, speed_delay=0.25)
            elif mood == 'angry':
                # fast spin
                _fast_spin(api)
            elif mood == 'sad':
                # nod/droop behaviour
                _nod(api)
            elif mood == 'tired':
                # slow left-right
                _left_right(api, distance=20, repeats=2, delay=0.7)
            elif mood == 'focused':
                # quick precise nudges
                _left_right(api, distance=15, repeats=3, delay=0.25)
            elif mood == 'agitated':
                # quick spin + nudge
                _spin(api, times=3, speed_delay=0.18)
                _left_right(api, distance=25, repeats=1, delay=0.2)
            else:
                # neutral / unknown -> small nudge
                _left_right(api, distance=10, repeats=1, delay=0.3)
        except Exception:
            pass

    t = threading.Thread(target=runner, name=f"mood-{mood}")
    t.daemon = True
    t.start()
    return t
