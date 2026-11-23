"""
Tests for the ESP32 firmware (ESP32_files/Cubotino_servos.py), run on the PC with stubs of the MicroPython
'machine' and 'utime' modules: the servo sleeps take no time, and the servo positions are recorded.

Run from the repo root:
    python3 -m unittest discover -s tests -v
"""

import contextlib
import importlib
import io
import os
import sys
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ESP32 = os.path.join(REPO, 'ESP32_files')
sys.path.insert(0, os.path.join(REPO, 'PC_files'))
sys.path.insert(0, HERE)

import Cubotino_moves as cm                             # noqa: E402
from test_cubotino_moves import solver_like_set         # noqa: E402


class FakePWM:
    def __init__(self, pin, freq=50):
        self.history = []
        self._duty = 0

    def duty(self, value=None):
        if value is None:
            return self._duty
        self._duty = value
        self.history.append(value)


class FakeTouchPad:
    def __init__(self, pin):
        pass

    def read(self):
        return 500                       # never below the stop threshold


def load_servos_module():
    """Imports Cubotino_servos.py with stubbed MicroPython modules, and no UART data."""
    machine = types.ModuleType('machine')
    machine.Pin = lambda *a, **k: None
    machine.PWM = FakePWM
    machine.TouchPad = FakeTouchPad
    utime = types.ModuleType('utime')
    utime.sleep = utime.sleep_ms = lambda t: None
    utime.time = lambda: 0
    sys.modules['machine'], sys.modules['utime'] = machine, utime
    sys.path.insert(0, ESP32)
    try:
        sys.modules.pop('Cubotino_servos', None)
        servo = importlib.import_module('Cubotino_servos')
    finally:
        sys.path.remove(ESP32)
    servo.select = types.SimpleNamespace(select=lambda r, w, x, t=0: ([], [], []))   # nothing on the UART
    return servo


class FakeUart:
    """Characters the PC has sent to the ESP32, read one at a time as MicroPython's sys.stdin does."""
    def __init__(self, text):
        self.text = text

    def read(self, n):
        ch, self.text = self.text[:n], self.text[n:]
        return ch

    def select(self, r, w, x, t=0):
        return ([self] if self.text else [], [], [])


def feed_uart(module, text):
    """Makes the module read text from the UART."""
    uart = FakeUart(text)
    module.sys = types.SimpleNamespace(stdin=uart)
    module.select = types.SimpleNamespace(select=uart.select)
    return uart


def load_main_functions(servo):
    """Runs ESP32_files/main.py up to its main program (the infinite loop), and returns its functions."""
    with open(os.path.join(ESP32, 'main.py')) as f:
        source = f.read()
    source = source[:source.index('# MAIN PROGRAM')]
    sys.modules['Cubotino_servos'] = servo               # imported by main.py functions
    machine = sys.modules['machine']
    machine.Timer = lambda *a, **k: None
    namespace = {'__name__': 'esp32_main'}
    exec(compile(source, 'main.py', 'exec'), namespace)
    namespace['debug'] = False
    return namespace


class TestFirmware(unittest.TestCase):

    def setUp(self):
        self.servo = load_servos_module()
        self.cwd = os.getcwd()
        os.chdir(ESP32)                                  # the firmware reads Cubotino_settings.txt from its folder
        self.addCleanup(os.chdir, self.cwd)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(self.servo.init_servo())

    def solve(self, moves):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status, _ = self.servo.servo_solve_cube(moves, False, FakeTouchPad(None), 250)
        return status, out.getvalue()

    def test_check_moves_accepts_planner_moves(self):
        for sol in solver_like_set(seed=11, count=200):
            _, moves, tot = cm.robot_required_moves(sol, '')
            self.assertEqual(self.servo.check_moves(moves, False), (True, tot), moves)

    def test_check_moves_rejects_out_of_range(self):
        for moves in ('R1R1', 'R3R3', 'S1S1', 'S3R3', 'R1F1S1', 'S3S1S3S3'):
            self.assertFalse(self.servo.check_moves(moves, False)[0], moves)

    def test_out_of_range_moves_are_rejected_before_moving(self):
        b_history = len(self.servo.b_servo.history)
        status, out = self.solve('R1R1')
        self.assertEqual(status, 'Robot_stopped')
        self.assertIn('servo range error', out)
        self.assertNotIn('i_', out)                      # no progress: no move was made
        self.assertNotIn(self.servo.b_servo_CW, self.servo.b_servo.history[b_history:])

    def test_planner_moves_are_executed(self):
        _, moves, tot = cm.robot_required_moves('U1 R2 F3 D2 L1 B2', '')
        status, out = self.solve(moves)
        self.assertEqual(status, 'Cube_solved')
        self.assertEqual(out.count('i_'), tot)          # one progress message per servo move, as the GUI expects

    def test_stop_request_is_matched_exactly(self):
        cases = {'[stop]\n': True, '[stop]\r\n': True, '[led_off]\n[stop]\n': True,
                 '[nonstop]\n': False, '[stopped]\n': False, '[start]\n': False, '': False}
        for text, stops in cases.items():
            self.servo.stop_servos = False
            feed_uart(self.servo, text)
            self.assertEqual(bool(self.servo.check_uart(False, FakeTouchPad(None), 250)), stops, repr(text))


class TestFirmwareMain(unittest.TestCase):

    def setUp(self):
        self.servo = load_servos_module()
        self.cwd = os.getcwd()
        os.chdir(ESP32)
        self.addCleanup(os.chdir, self.cwd)
        self.main = load_main_functions(self.servo)

    def test_parse_command(self):
        parse = self.main['parse_command']
        self.assertEqual(parse('start'), ('start', ''))
        self.assertEqual(parse('led_on\r'), ('led_on', ''))
        self.assertEqual(parse('test(ccw)'), ('test', '(ccw)'))
        self.assertEqual(parse('new_settings(1,2,3)'), ('new_settings', '(1,2,3)'))
        self.assertEqual(parse('restart'), ('restart', ''))      # not 'start'

    def test_servo_tests_are_matched_exactly(self):
        calls = []
        self.servo.flip_test = lambda: calls.append('flip')
        self.servo.close_cover = lambda: calls.append('close')
        self.servo.open_cover = lambda: calls.append('open')
        self.servo.rotate_CCW_CW_test = lambda d: calls.append(d)
        self.servo.rotate_home_test = lambda: calls.append('home')
        expected = {'(flip)': ['flip'], '(close)': ['close'], '(open)': ['open'], '(ccw)': ['CCW'],
                    '(home)': ['home'], '(cw)': ['CW'], '(xcw)': [], '(reopen)': [], '': []}
        for data, call in expected.items():
            calls.clear()
            with contextlib.redirect_stdout(io.StringIO()):
                self.main['test_robot'](data, False)
            self.assertEqual(calls, call, data)


if __name__ == '__main__':
    unittest.main()
