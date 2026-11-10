"""
#############################################################################################################
# Andrea Favero April 2022
#
# From Kociemba solver to robot moves
# This applies to "CUBOTino", a simpler (and much cheaper) Rubik's cube solver robot than my first one:
# (https://www.youtube.com/watch?v=oYRXe4NyJqs)
#
# The Kociemba cube solver returns a solution that has to be translated into robot movements
# Notations for faces are as per URF notations: U=Up, R=Right, F=Front, D=Down, L=Left, B=Back
# Tipically the solver considers the cube orientation to don't change while solving it....
# On this simpler robot, the mechanical constraints (servo rotation limited to 180deg), suggests a different approach:
#  - The face to be turned, according to the solver, will be adapted to reflect the side such face is located
#  - This means a supposed R1 might change L1 if the left cube side is located on the right side at that moment in time
#
#
# Possible moves with this robot
# 1) Spins the complete cube ("S") laying on the bottom face: 1 means CW 90deg turns, while 3 means 90CCW turn
# 2) Flips the complete cube ("F") by "moving" the Front face to Bottom face: Only positive values are possible
# 3) Rotates the bottom layer ("R") while costraining the 2nd and 3rd layer.
# 4) The order of S, F has to be strictly followed
# 5) Example 'F1R1S3' means: 1x cube Flip, 1x (90deg) CW rotation of the 1st (Down) layer, 1x (90deg) CCW cube Spin
#
#
# Optimized planner (replaces the original fixed lookup table):
# The bottom servo has only three positions (CCW, home, CW), and both S and R moves step it by one position.
# The original table forced the cube holder back to home after every solver move, which often required
# extra flips (i.e. L1 = 'S3F3R1', with 3 flips) and extra "re-homing" spins.
# Here the whole solution is planned at once, as a shortest path (Dijkstra) over the robot states:
#   (solver move index, cube orientation, holder position, quarter turns done on the faces of that move)
# The cost of each robot action is its duration with the default servo timings, so the
# planner minimizes the solving time, not just the movements count.
# Consecutive solver moves on opposite faces (i.e. 'U1 D2') commute, so they are planned as a single
# group, letting the planner pick the cheapest order.
#
#
# Note: If the cube status has been entered manually, or detected via a webcam apart from the robot,
# the cube should be positioned with the Front face facing the viewer, and Upper facing upward
#
#############################################################################################################
"""

# Duration (ms) of each robot action, from the default servo timings in Cubotino_settings.txt
# (multiples of 100ms: the planner works in 100ms units)
FLIP_COST = 1600     # flipper up (800) + back down (800)
SPIN_COST = 1100     # b_spin_time
ROTATE_COST = 1900   # cover close (300) + b_rotate_time (1200) + b_rel_time (100) + cover open (300)
ROTATE_AFTER_FLIP_COST = 1700   # after a flip the cover closes straight from the flip position (+100), and the
                                # flip doesn't go back to open (-800 +800 already in FLIP_COST, -300 cover close)


# Cube orientation is a 6 chars string: the face located at the positions U, D, F, B, L, R (in this order)
START_ORIENTATION = 'UDFBLR'
"""
               _______
              |       |
              | o[0]U |
       _______|_______|_______ _______
      |       |       |       |       |
      | o[4]L | o[2]F | o[5]R | o[3]B |
      |_______|_______|_______|_______|
              |       |
              | o[1]D |
              |_______|
"""

def flip_effect(o):
    """ Cube orientation after a single Flip: the Front face moves to the Bottom, L and R don't change."""
    return o[3] + o[2] + o[0] + o[1] + o[4] + o[5]


def spinCW_effect(o):
    """ Cube orientation after a single CW spin (S1): the Left face moves to the Front, U and D don't change."""
    return o[0] + o[1] + o[4] + o[5] + o[3] + o[2]


def spinCCW_effect(o):
    """ Cube orientation after a single CCW spin (S3): the Right face moves to the Front, U and D don't change."""
    return o[0] + o[1] + o[5] + o[4] + o[2] + o[3]




def opp_face(face):
    """ This function returns the opposite face of the one in argument."""
    return {'F':'B', 'B':'F', 'U':'D', 'D':'U', 'R':'L', 'L':'R'}.get(face, 'Error')




def group_solution(solution):
    """ Splits the solver string (spaces already removed) into groups of commuting moves.
        Each group is a tuple of (face, quarter turns) pairs: a single move, or two moves on opposite faces."""

    moves = [(solution[i], int(solution[i+1]) % 4) for i in range(0, len(solution) - 1, 2)]
    groups = []
    i = 0
    while i < len(moves):
        if i + 1 < len(moves) and moves[i+1][0] == opp_face(moves[i][0]):
            groups.append((moves[i], moves[i+1]))
            i += 2
        else:
            groups.append((moves[i],))
            i += 1
    return groups




def _build_orientation_tables():
    """ Enumerates the 24 cube orientations, and tabulates the orientation index after a flip / CW spin / CCW spin,
        and the face located at the bottom, for each orientation index."""

    orients = [START_ORIENTATION]
    index = {START_ORIENTATION: 0}
    i = 0
    while i < len(orients):
        for effect in (flip_effect, spinCW_effect, spinCCW_effect):
            o = effect(orients[i])
            if o not in index:
                index[o] = len(orients)
                orients.append(o)
        i += 1
    flip = bytes([index[flip_effect(o)] for o in orients])
    spin_cw = bytes([index[spinCW_effect(o)] for o in orients])
    spin_ccw = bytes([index[spinCCW_effect(o)] for o in orients])
    bottom = ''.join([o[1] for o in orients])
    return flip, spin_cw, spin_ccw, bottom

_FLIP, _SPIN_CW, _SPIN_CCW, _BOTTOM = _build_orientation_tables()

# A robot state is s = orientation_index*3 + holder, with holder 0=CCW, 1=home, 2=CW
_N_STATES = 72
_START_STATE = 1                                  # starting orientation, holder at home
_INF = 1000000
# Robot actions of the search edges; a flip directly followed by a rotation is a single edge,
# as it is cheaper than a flip followed by anything else (ROTATE_AFTER_FLIP_COST)
_ACTIONS = ('F1', 'S1', 'S3', 'R1', 'R3', 'F1R1', 'F1R3')




def _search(group, sources):
    """ Dijkstra (bucket queue, so it is deterministic and cheap on MicroPython) within a single group of solver moves.
        Search node u = state*P + progress, where progress counts the quarter turns done (mod 4) on each face of the group.
        sources is a list of (cost, state) the search starts from.
        Returns the cost, parent node, action and starting state of each node, plus the target progress."""

    f_cost, s_cost, r_cost = FLIP_COST // 100, SPIN_COST // 100, ROTATE_COST // 100
    fr_cost = (FLIP_COST + ROTATE_AFTER_FLIP_COST) // 100
    P = 4 ** len(group)                           # number of progress values (4 or 16)
    target = 0
    for k in range(len(group)):
        target += group[k][1] * 4 ** k

    size = _N_STATES * P
    dist = [_INF] * size
    par = [-1] * size
    act = bytearray(size)
    orig = bytearray(size)
    buckets = {}
    for c, s in sources:
        u = s * P
        if c < dist[u]:
            dist[u] = c
            orig[u] = s
            if c in buckets:
                buckets[c].append(u)
            else:
                buckets[c] = [u]

    c = min(buckets) if buckets else 0
    while buckets:
        nodes = buckets.pop(c, None)
        if nodes:
            for u in nodes:
                if dist[u] != c:                  # node already reached with a lower cost
                    continue
                s, prog = divmod(u, P)
                if prog == target:                # group done: this is an exit node
                    continue
                oi, h = divmod(s, 3)
                oi_flip = _FLIP[oi]
                steps = [(oi_flip * 3 + h, prog, 0, f_cost)]
                if h < 2:
                    steps.append((_SPIN_CW[oi] * 3 + h + 1, prog, 1, s_cost))
                if h > 0:
                    steps.append((_SPIN_CCW[oi] * 3 + h - 1, prog, 2, s_cost))
                for k in range(len(group)):
                    w = 4 ** k
                    q = (prog // w) % 4
                    for o2, a, cost in ((oi, 3, r_cost), (oi_flip, 5, fr_cost)):   # rotation, flip + rotation
                        if _BOTTOM[o2] == group[k][0]:      # a face of this group is on the bottom
                            if h < 2:
                                steps.append((o2 * 3 + h + 1, prog + ((q + 1) % 4 - q) * w, a, cost))
                            if h > 0:
                                steps.append((o2 * 3 + h - 1, prog + ((q + 3) % 4 - q) * w, a + 1, cost))
                for s2, prog2, a, step_cost in steps:
                    u2 = s2 * P + prog2
                    c2 = c + step_cost
                    if c2 < dist[u2]:
                        dist[u2] = c2
                        par[u2] = u
                        act[u2] = a
                        orig[u2] = orig[u]
                        if c2 in buckets:
                            buckets[c2].append(u2)
                        else:
                            buckets[c2] = [u2]
        c += 1
    return dist, par, act, orig, P, target




def plan_moves(groups):
    """ Shortest (in time) sequence of robot actions executing all the groups of solver moves.
        Forward pass: for each group, the cheapest cost (and the entry state it came from) to finish it in each state.
        Backward pass: the actions within each group are re-generated from the chosen entry state.
        Returns a list (one entry per group) of lists of robot actions, i.e. [['S1', 'F1', 'R1'], ...]."""

    entry = [(0, _START_STATE)]
    backs = []
    for group in groups:
        dist, par, act, orig, P, target = _search(group, entry)
        back = bytearray(_N_STATES)
        entry = []
        for s in range(_N_STATES):
            u = s * P + target
            if dist[u] < _INF:
                entry.append((dist[u], s))
                back[s] = orig[u]
        backs.append(back)

    blocks = [[] for _ in groups]
    if not groups:
        return blocks
    s_out = min(entry)[1]                         # cheapest final state
    for g in range(len(groups) - 1, -1, -1):
        s_in = backs[g][s_out]
        dist, par, act, orig, P, target = _search(groups[g], [(0, s_in)])
        u = s_out * P + target
        while u != s_in * P:
            a = _ACTIONS[act[u]]
            for i in range(len(a) - 2, -1, -2):  # 'F1R1' is added as 'F1', 'R1'
                blocks[g].insert(0, a[i:i+2])
            u = par[u]
        s_out = s_in
    return blocks




def join_actions(actions):
    """ Joins a list of robot actions into the robot moves string, merging consecutive flips (i.e. F1F1 -> F2)."""

    out = []
    for a in actions:
        if a == 'F1' and out and out[-1][0] == 'F':
            out[-1] = 'F' + str(int(out[-1][1]) + 1)
        else:
            out.append(a)
    return ''.join(out)




def count_moves(moves):
    """Counts the total amount of robot movements."""

    robot_tot_moves = 0               # counter for all the robot movements
    for i in range(len(moves)):       # iterates over the string of robot movements
        if moves[i] == 'F':           # case there is a cube flip on robot movements
            robot_tot_moves+=int(moves[i+1])   # increases the total amount of robot movements by the amount of flips
        elif moves[i] == 'R' or moves[i] == 'S':   # case there is a layer rotation or a cube spin
            robot_tot_moves+=1        # increases by 1 (cannot be more) the total amount of robot movements
    return robot_tot_moves            # total amount of robot moves is returned




def robot_required_moves(solution, solution_Text):
    """ This function converts the cube manouvre from Kociemba solver string into the robot movements.
        Returns a dict with the robot moves per solver move (or commuting pair of moves), the string
        with all the robot moves, and the total amount of robot movements."""

    solution=solution.strip()                     # eventual empty spaces are removed from the string
    solution=solution.replace(" ", "")            # eventual empty spaces are removed from the string
    robot={}                                      # empty dict to store all the robot moves
    moves=''                                      # empty string to store all the robot moves
    robot_tot_moves = 0                           # counter for all the robot movements

    if solution_Text != 'Error':                  # case the solver did not return an error
        groups = group_solution(solution)         # solver moves, grouped when commuting (opposite faces)
        blocks = plan_moves(groups)               # optimal robot actions per group
        for block, actions in enumerate(blocks):
            robot[block] = join_actions(actions)  # robot movements dict is updated
        moves = join_actions([a for actions in blocks for a in actions])   # robot movements string
        robot_tot_moves = count_moves(moves)      # counter for the total amount of robot movements

    return robot, moves, robot_tot_moves  # returns a dict with all the robot moves, string with all the moves and total robot movements





if __name__ == "__main__":
    """ This function convert the cube solution string 'U2 L1 R1 D2 B2 R1 D2 B2 D2 L3 B3 R3 F2 D3 L1 U2 F2 D3 B3 D1' in robot moves
        Robot moves are printed on the REPL
        Robot moves are translated to servo moves: Initially are print per ach of the cube solving string manoeuvre
        Afterward all the strings are combined in a single string, for the Cubotino_servo.py module to control the servos."""  
    
    print()
    print("Example of robot movements for solver solution: 'U2 L1 R1 D2 B2 R1 D2 B2 D2 L3 B3 R3 F2 D3 L1 U2 F2 D3 B3 D1'")
    print("Robot moves are notated with the 3 letters S, F, R (Spin, Flip, Rotate) followed by a number")
    print("Number '1' for S ans R identifies CW rotation, by loking to the bottom face, while number '3' stands for CCW")
    print("Example 'F1R1S3' means: 1x cube Flip, 1x (90deg) CW rotation of the 1st (bottom) layer, 1x (90deg) CCW cube Spin")
    print()
    
    solution = 'U2 L1 R1 D2 B2 R1 D2 B2 D2 L3 B3 R3 F2 D3 L1 U2 F2 D3 B3 D1'
#     solution = 'U2 D2 R2 L2 F2 B2'
#     solution = 'R2 L1 D3 F2 L2 B1 L1 U3 R1 F1 L2 D3 F2 D1 F2 B2 D2'
    
    solution_Text = ""
    robot, moves, robot_tot_moves = robot_required_moves(solution, solution_Text)
    print(f'\nnumber of robot movements: {robot_tot_moves}')
    
    print()    
    print(f'robot movements: ')
    
    servo_moves=""
    for step, movements in robot.items():
        print(f'step:{step}, robot moves:{movements}')
        servo_moves+=moves
    
    print(f'\nstring command to the robot servos driver: {moves}\n')
    
    
    
    
    ######################################              imports and hardware settings, for the robot activation part
    
    import Cubotino_servos as servo
    from machine import Pin, TouchPad
    from utime import time
    
    stop_btn=TouchPad(Pin(32))          # create the touch pad button object in GPIO 13
    stop_btn_value=stop_btn.read()      # create a global variable for the touchPad button variable
    btn_ref = 0                         # button threshold reference is set initially to zero
    n=1000                              # number of touch buttong reading to perform as initial reference
    for i in range(n):                  # iteration for 1000 readings
        btn_ref+=stop_btn_value         # touch pad value is added to the reference at each iteration                                
    btn_ref=btn_ref//n//2               # button threshold reference is set at 50% the average value during previous iteration
        
    debug=False                         # boolean variable to forse some prints for debug purpose
    
    if servo.init_servo(debug):                        # servos are initialized
        start=time()                                   # time before start the robot movements
        robot_status=servo.servo_solve_cube(moves, debug, stop_btn, btn_ref)   # robot solver is called
        end=time()                                     # time after the robot movements

        if robot_status == 'Cube_solved':              # case the robot solver returns Cube_solved in the robot_status
            print("\n Cube is solved")                 # print the status as feedback
            print(f' Solving time: {end-start} secs')  # print the solving time as feedback
        
        elif robot_status == 'Robot_stopped':          # case the robot solver returns Robot_stopped in the robot_status
            print("\n Robot has been stopped")         # print the status as feedback
        
        servo.init_servo(debug)
        
    ######################################
