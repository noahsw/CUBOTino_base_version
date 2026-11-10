"""
Test helpers: a physical Rubik's cube model, and an emulator of the ESP32 firmware (Cubotino_servos.servo_solve_cube).

The cube model is independent from Cubotino_moves.py: the cube is 54 stickers in 3D space, and every robot
action is a geometric rotation. So a planner bug (wrong face tracking, wrong direction, servo out of range)
shows up as a cube that is not solved at the end.

Robot frame: x = Right, y = Up, z = Front (looking at the robot from the front).
  - Flip (F1): the whole cube turns +90deg around x, the Front face goes to the Bottom
  - Spin CW (S1): the whole cube turns +90deg around y, the Front face goes to the Right
  - Rotate CW (R1): only the bottom layer turns +90deg around y (= clockwise looking at the bottom face)
These conventions are anchored by the legacy planner (the original lookup table, known to work on the robot):
it must solve the cube in this simulator too (see test_legacy_planner_solves_in_simulator).
"""

import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FACE_NORMALS = {'U': (0, 1, 0), 'D': (0, -1, 0), 'R': (1, 0, 0),
                'L': (-1, 0, 0), 'F': (0, 0, 1), 'B': (0, 0, -1)}


def _rot(v, axis, k):
    """Rotates vector v by k*90deg (right hand rule) around the axis 'x', 'y' or 'z'."""
    x, y, z = v
    for _ in range(k % 4):
        if axis == 'x':
            x, y, z = x, -z, y
        elif axis == 'y':
            x, y, z = z, y, -x
        else:
            x, y, z = -y, x, z
    return (x, y, z)


class Cube:
    """54 stickers: (cubie position, sticker normal, color). The color is the face name it belongs to."""

    def __init__(self):
        self.stickers = []
        for color, n in FACE_NORMALS.items():
            for a in (-1, 0, 1):
                for b in (-1, 0, 1):
                    axis = [i for i in range(3) if n[i] != 0][0]
                    others = [i for i in range(3) if i != axis]
                    pos = [0, 0, 0]
                    pos[axis] = n[axis]
                    pos[others[0]] = a
                    pos[others[1]] = b
                    self.stickers.append((tuple(pos), n, color))

    def rotate(self, axis, k, layer=None):
        """Rotates the stickers (all, or those with the given axis coordinate) by k*90deg around axis."""
        i = 'xyz'.index(axis)
        new = []
        for pos, n, color in self.stickers:
            if layer is None or pos[i] == layer:
                pos, n = _rot(pos, axis, k), _rot(n, axis, k)
            new.append((pos, n, color))
        self.stickers = new

    def solver_move(self, face, q):
        """Kociemba notation move (cube not re-oriented): face turned q*90deg clockwise, looking at that face."""
        n = FACE_NORMALS[face]
        i = [j for j in range(3) if n[j] != 0][0]
        # clockwise looking at the face = -90deg around its outward normal
        self.rotate('xyz'[i], -q * n[i], layer=n[i])

    def scramble_for(self, solution):
        """Applies the inverse of the solution, so that applying the solution solves the cube."""
        moves = solution.replace(' ', '')
        for j in range(len(moves) - 2, -1, -2):
            self.solver_move(moves[j], 4 - int(moves[j + 1]))

    def is_solved(self):
        """Every face (whatever orientation the cube has on the robot) has a single color."""
        faces = {}
        for pos, n, color in self.stickers:
            faces.setdefault(n, set()).add(color)
        return len(faces) == 6 and all(len(c) == 1 for c in faces.values())


def load_settings(path=os.path.join(REPO, 'ESP32_files', 'Cubotino_settings.txt')):
    """Servo timings (ms) from the robot settings file, with the same indexes as Cubotino_servos.upload_settings."""
    with open(path) as f:
        data = f.read().strip().strip('()').split(',')
    s = [int(x) for x in data[:16]]
    return {'flip_to_close': s[4], 'close_to_flip': s[5], 'flip_open': s[6], 'open_close': s[7],
            'spin': s[13], 'rotate': s[14], 'rel': s[15]}


class FirmwareResult:
    def __init__(self):
        self.time_ms = 0         # servo time, as per the sleep_ms calls of the firmware
        self.servo_moves = 0     # moves reported to the GUI progress bar (update_moves calls)
        self.anomalies = []      # robot moves the firmware skipped or executed differently than written


def run_firmware(moves, cube=None, settings=None):
    """Executes the robot moves string with the same logic of Cubotino_servos.servo_solve_cube,
    applying the physical effect of each servo action to the cube."""
    t = settings or load_settings()
    cube = cube or Cube()
    res = FirmwareResult()
    holder = 'home'          # 'home', 'CW', 'CCW'
    cover = 'open'           # 'open', 'close', 'flip'
    operable = True          # b_servo_operable

    def flip_to_close():
        nonlocal cover, operable
        res.time_ms += t['flip_to_close'] if cover == 'flip' else t['open_close']
        cover, operable = 'close', True

    def open_cover():
        nonlocal cover, operable
        res.time_ms += t['open_close']
        cover, operable = 'open', True

    n = len(moves)
    for i in range(n):
        c = moves[i]
        if c == 'F':
            flips = int(moves[i + 1])
            for f in range(flips):
                res.time_ms += t['close_to_flip'] if cover == 'close' else t['flip_open']
                cover, operable = 'flip', False
                cube.rotate('x', 1)
                res.servo_moves += 1
                if f < flips - 1:
                    res.time_ms += t['flip_open']
                    cover, operable = 'open', True
                if f == flips - 1 and n - (i + 2) > 0:
                    if moves[i + 2] == 'R':
                        flip_to_close()
                    elif moves[i + 2] == 'S':
                        res.time_ms += t['flip_open']
                        cover, operable = 'open', True

        elif c == 'S':
            direction = 'CCW' if int(moves[i + 1]) == 3 else 'CW'
            if holder == 'home':
                if not operable:                       # spin_out does nothing
                    res.anomalies.append((i, 'spin skipped, bottom servo not operable'))
                    continue
                cube.rotate('y', 1 if direction == 'CW' else -1)
                holder = direction
                res.time_ms += t['spin']
                res.servo_moves += 1
            else:
                if not operable:                       # spin_home opens the cover instead
                    res.anomalies.append((i, 'spin replaced by open_cover'))
                    open_cover()
                    res.servo_moves += 1
                    continue
                if holder == direction:
                    res.anomalies.append((i, 'spin direction ignored: holder goes back home'))
                cube.rotate('y', -1 if holder == 'CW' else 1)
                holder = 'home'
                res.time_ms += t['spin']
                res.servo_moves += 1

        elif c == 'R':
            direction = 'CCW' if int(moves[i + 1]) == 3 else 'CW'
            if holder != 'home' and holder == direction:
                res.anomalies.append((i, 'rotation skipped, servo already at ' + holder))
                continue
            if cover != 'close':
                flip_to_close()
            cube.rotate('y', 1 if direction == 'CW' else -1, layer=-1)
            holder = direction if holder == 'home' else 'home'
            res.time_ms += t['rotate'] + t['rel']
            open_cover()
            res.servo_moves += 1
    return res
