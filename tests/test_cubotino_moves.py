"""
Regression tests for Cubotino_moves.py (solver solution -> robot moves).

Run from the repo root (only the Python standard library is needed):
    python3 -m unittest discover -s tests -v

What is covered:
  - correctness: the robot moves, executed by an emulator of the ESP32 firmware on a physical cube model,
    solve every scrambled cube (single moves, opposite face pairs, random solver-like and arbitrary sequences)
  - firmware compatibility: no robot move is skipped or altered by the firmware (servo range, directions),
    and the move count matches the progress messages the firmware sends to the GUI
  - optimality: the planner cost always equals the one from a simple reference Dijkstra
  - no regression vs the original lookup table planner (tests/legacy_Cubotino_moves.py, a frozen copy):
    never slower on any solution, and overall at least 20% faster
  - API and string format expected by the GUI and the firmware, determinism, PC/ESP32 copies identical
"""

import ast
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, 'PC_files'))
sys.path.insert(0, HERE)

import Cubotino_moves as cm                       # noqa: E402
import legacy_Cubotino_moves as legacy            # noqa: E402
import reference_planner as ref                   # noqa: E402
from robot_sim import Cube, run_firmware          # noqa: E402

AXIS = {'U': 'y', 'D': 'y', 'F': 'z', 'B': 'z', 'L': 'x', 'R': 'x'}
ALL_MOVES = [f + q for f in 'UDFBLR' for q in '123']


def solver_like_solution(rng, length):
    """Random solution shaped like the Kociemba solver output: no face twice in a row,
    at most two consecutive moves on the same axis (i.e. 'U1 D2' but never 'U1 D2 U3')."""
    sol = []
    while len(sol) < length:
        m = rng.choice(ALL_MOVES)
        if sol and sol[-1][0] == m[0]:
            continue
        if len(sol) >= 2 and AXIS[sol[-1][0]] == AXIS[m[0]] == AXIS[sol[-2][0]]:
            continue
        sol.append(m)
    return ' '.join(sol)


def solver_like_set(seed=1, count=300):
    rng = random.Random(seed)
    return [solver_like_solution(rng, rng.randint(1, 24)) for _ in range(count)]


SAMPLE_SOLUTIONS = [
    'U2 L1 R1 D2 B2 R1 D2 B2 D2 L3 B3 R3 F2 D3 L1 U2 F2 D3 B3 D1',   # example in Cubotino_moves.py
    'U2 D2 R2 L2 F2 B2',
    'R2 L1 D3 F2 L2 B1 L1 U3 R1 F1 L2 D3 F2 D1 F2 B2 D2',
]


class PlannerTestCase(unittest.TestCase):

    def assert_solves(self, planner, solution):
        """The robot moves for solution solve the scrambled cube, with no firmware anomaly."""
        robot, moves, tot = planner.robot_required_moves(solution, '')
        cube = Cube()
        cube.scramble_for(solution)
        res = run_firmware(moves, cube)
        self.assertEqual(res.anomalies, [], f'firmware would not execute {moves!r} as written ({solution!r})')
        self.assertTrue(cube.is_solved(), f'cube not solved by {moves!r} for solution {solution!r}')
        return robot, moves, tot, res


class TestSimulator(PlannerTestCase):
    """The test harness itself must be trustworthy."""

    def test_solver_move_and_inverse(self):
        for m in ALL_MOVES:
            cube = Cube()
            cube.solver_move(m[0], int(m[1]))
            self.assertFalse(cube.is_solved(), m)
            cube.solver_move(m[0], 4 - int(m[1]))
            self.assertTrue(cube.is_solved(), m)

    def test_four_quarter_turns_are_identity(self):
        for face in 'UDFBLR':
            cube = Cube()
            for _ in range(4):
                cube.solver_move(face, 1)
            self.assertTrue(cube.is_solved())

    def test_unsolved_cube_detected(self):
        cube = Cube()
        cube.scramble_for(SAMPLE_SOLUTIONS[0])
        self.assertFalse(cube.is_solved())

    def test_legacy_planner_solves_in_simulator(self):
        """The original lookup table works on the real robot: if it solves the cube here,
        the simulator conventions (flip/spin/rotate directions) match the robot."""
        for sol in ALL_MOVES + SAMPLE_SOLUTIONS + solver_like_set(seed=7, count=100):
            self.assert_solves(legacy, sol)

    def test_firmware_emulator_flags_invalid_strings(self):
        self.assertTrue(run_firmware('R1R1').anomalies)   # 2nd CW rotation beyond the servo range
        self.assertTrue(run_firmware('S1S1').anomalies)   # spin direction ignored at the CW end
        self.assertEqual(run_firmware('R1S3R3S1').anomalies, [])


class TestCorrectness(PlannerTestCase):

    def test_every_single_move(self):
        for m in ALL_MOVES:
            with self.subTest(move=m):
                self.assert_solves(cm, m)

    def test_every_opposite_face_pair(self):
        for a in 'UDFBLR':
            b = cm.opp_face(a)
            for qa in '123':
                for qb in '123':
                    with self.subTest(pair=a + qa + b + qb):
                        self.assert_solves(cm, f'{a}{qa} {b}{qb}')

    def test_every_two_move_sequence(self):
        for m1 in ALL_MOVES:
            for m2 in ALL_MOVES:
                with self.subTest(seq=m1 + m2):
                    self.assert_solves(cm, f'{m1} {m2}')

    def test_sample_solutions(self):
        for sol in SAMPLE_SOLUTIONS:
            with self.subTest(solution=sol):
                self.assert_solves(cm, sol)

    def test_random_solver_like_solutions(self):
        for sol in solver_like_set(seed=1, count=300):
            with self.subTest(solution=sol):
                self.assert_solves(cm, sol)

    def test_arbitrary_sequences(self):
        """Not produced by the solver (same face repeated, 3+ moves on an axis), but must still be correct."""
        rng = random.Random(3)
        for _ in range(150):
            sol = ' '.join(rng.choice(ALL_MOVES) for _ in range(rng.randint(1, 12)))
            with self.subTest(solution=sol):
                self.assert_solves(cm, sol)

    def test_long_solution(self):
        self.assert_solves(cm, solver_like_solution(random.Random(5), 30))


class TestFirmwareCompatibility(PlannerTestCase):

    def test_format(self):
        """Only S1/S3/R1/R3/F1-F3 tokens, flips merged (never 'F1F1')."""
        for sol in SAMPLE_SOLUTIONS + solver_like_set(seed=2, count=100):
            _, moves, _ = cm.robot_required_moves(sol, '')
            self.assertRegex(moves, r'^([SR][13]|F[123])*$')
            self.assertNotRegex(moves, r'F\dF')

    def test_move_count_matches_firmware_progress(self):
        """The GUI progress bar expects as many servo moves as reported by the firmware."""
        for sol in SAMPLE_SOLUTIONS + solver_like_set(seed=3, count=100):
            _, moves, tot, res = self.assert_solves(cm, sol)
            self.assertEqual(tot, res.servo_moves, sol)
            self.assertEqual(tot, cm.count_moves(moves), sol)

    def test_servo_angle_within_range(self):
        """Same rule as Cubotino_servos.check_moves, but with the real range (-90..+90 deg)."""
        for sol in solver_like_set(seed=4, count=200):
            _, moves, _ = cm.robot_required_moves(sol, '')
            angle = 0
            for i in range(0, len(moves), 2):
                if moves[i] in 'SR':
                    angle += 90 if moves[i + 1] == '1' else -90
                    self.assertTrue(-90 <= angle <= 90, (sol, moves, i))


class TestOptimality(unittest.TestCase):

    def test_matches_reference_search(self):
        for sol in SAMPLE_SOLUTIONS + solver_like_set(seed=5, count=60):
            with self.subTest(solution=sol):
                _, moves, _ = cm.robot_required_moves(sol, '')
                groups = cm.group_solution(sol.replace(' ', ''))
                self.assertEqual(ref.model_cost(cm, moves), ref.optimal_cost(cm, groups))

    def test_cost_model_matches_firmware_timing(self):
        """The planner costs are the real servo times with the default settings: if the firmware timing logic
        or the settings change, the cost constants in Cubotino_moves.py need to be updated too."""
        for sol in SAMPLE_SOLUTIONS + solver_like_set(seed=10, count=100):
            for planner in (cm, legacy):
                _, moves, _ = planner.robot_required_moves(sol, '')
                self.assertEqual(ref.model_cost(cm, moves), run_firmware(moves).time_ms, moves)

    def test_grouping_only_pairs_opposite_faces(self):
        for sol in solver_like_set(seed=6, count=100):
            groups = cm.group_solution(sol.replace(' ', ''))
            self.assertEqual([m for g in groups for m in g],
                             [(m[0], int(m[1])) for m in sol.split()])
            for g in groups:
                self.assertIn(len(g), (1, 2))
                if len(g) == 2:
                    self.assertEqual(g[1][0], cm.opp_face(g[0][0]))


class TestNoRegressionVsLegacy(PlannerTestCase):

    def test_never_slower_than_legacy(self):
        new_total = old_total = 0
        for sol in ALL_MOVES + SAMPLE_SOLUTIONS + solver_like_set(seed=8, count=300):
            _, new_moves, _, new_res = self.assert_solves(cm, sol)
            _, old_moves, _, old_res = self.assert_solves(legacy, sol)
            self.assertLessEqual(new_res.time_ms, old_res.time_ms,
                                 f'{sol!r}: new {new_moves} slower than legacy {old_moves}')
            new_total += new_res.time_ms
            old_total += old_res.time_ms
        self.assertLessEqual(new_total, 0.8 * old_total,
                             f'expected >=20% faster than legacy, got {100 * (1 - new_total / old_total):.1f}%')


class TestApi(unittest.TestCase):

    def test_empty_solution(self):
        self.assertEqual(cm.robot_required_moves('', ''), ({}, '', 0))

    def test_solver_error(self):
        self.assertEqual(cm.robot_required_moves('U1', 'Error'), ({}, '', 0))

    def test_spaces_ignored(self):
        """The GUI removes the spaces, the ESP32 receives them: same result either way."""
        sol = SAMPLE_SOLUTIONS[0]
        self.assertEqual(cm.robot_required_moves(sol, ''), cm.robot_required_moves(sol.replace(' ', ''), ''))
        self.assertEqual(cm.robot_required_moves(' ' + sol + ' ', ''), cm.robot_required_moves(sol, ''))

    def test_robot_dict_concatenates_to_moves(self):
        for sol in SAMPLE_SOLUTIONS + solver_like_set(seed=9, count=50):
            robot, moves, _ = cm.robot_required_moves(sol, '')
            self.assertEqual(list(robot), list(range(len(robot))))
            self.assertEqual(''.join(robot[k] for k in sorted(robot)), moves)
            self.assertEqual(len(robot), len(cm.group_solution(sol.replace(' ', ''))))

    def test_deterministic(self):
        """The GUI and the ESP32 compute the moves separately: the progress bar relies on identical results."""
        for sol in SAMPLE_SOLUTIONS:
            self.assertEqual(cm.robot_required_moves(sol, ''), cm.robot_required_moves(sol, ''))

    def test_planning_is_repeatable_after_previous_runs(self):
        """No state leaks between runs (the robot can solve several cubes in one session)."""
        first = cm.robot_required_moves(SAMPLE_SOLUTIONS[1], '')
        cm.robot_required_moves(SAMPLE_SOLUTIONS[0], '')
        self.assertEqual(cm.robot_required_moves(SAMPLE_SOLUTIONS[1], ''), first)


class TestEsp32Copy(unittest.TestCase):

    def test_pc_and_esp32_modules_identical(self):
        with open(os.path.join(REPO, 'PC_files', 'Cubotino_moves.py'), 'rb') as f:
            pc = f.read()
        with open(os.path.join(REPO, 'ESP32_files', 'Cubotino_moves.py'), 'rb') as f:
            esp = f.read()
        self.assertEqual(pc, esp, 'PC_files and ESP32_files Cubotino_moves.py differ')

    def test_no_imports_outside_main(self):
        """The module runs on MicroPython: no top level imports (the __main__ demo is for the ESP32 only)."""
        with open(os.path.join(REPO, 'ESP32_files', 'Cubotino_moves.py')) as f:
            tree = ast.parse(f.read())
        imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertEqual(imports, [])


if __name__ == '__main__':
    unittest.main()
