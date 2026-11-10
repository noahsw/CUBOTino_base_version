"""
Reference implementation of the planner search, used only by the tests.

It is a straightforward (slow, memory hungry) Dijkstra over the whole solution, with a heap and dict-based nodes,
so it is easy to read and check. The planner in Cubotino_moves.py is a leaner version (layer by layer, arrays,
bucket queue) written to fit the ESP32: both must always find the same minimum cost.
"""

import heapq


def optimal_cost(cm, groups):
    """Minimum total cost (ms, with the cost constants of the Cubotino_moves module cm) to execute the groups."""
    n = len(groups)
    if n == 0:
        return 0
    # node: (group index, orientation, holder -1/0/1, last action was a flip, quarter turns per group face)
    start = (0, cm.START_ORIENTATION, 0, False, (0,) * len(groups[0]))
    best = {start: 0}
    heap = [(0, start)]
    while heap:
        cost, node = heapq.heappop(heap)
        if cost > best[node]:
            continue
        g, o, p, lf, prog = node
        r_cost = cm.ROTATE_AFTER_FLIP_COST if lf else cm.ROTATE_COST
        if g == n:
            return cost
        group = groups[g]
        if all(prog[k] == group[k][1] for k in range(len(group))):
            steps = [((g + 1, o, p, lf, (0,) * len(groups[g + 1]) if g + 1 < n else ()), 0)]
        else:
            steps = [((g, cm.flip_effect(o), p, True, prog), cm.FLIP_COST)]
            if p < 1:
                steps.append(((g, cm.spinCW_effect(o), p + 1, False, prog), cm.SPIN_COST))
            if p > -1:
                steps.append(((g, cm.spinCCW_effect(o), p - 1, False, prog), cm.SPIN_COST))
            for k in range(len(group)):
                if o[1] == group[k][0]:
                    if p < 1:
                        steps.append(((g, o, p + 1, False, prog[:k] + ((prog[k] + 1) % 4,) + prog[k + 1:]), r_cost))
                    if p > -1:
                        steps.append(((g, o, p - 1, False, prog[:k] + ((prog[k] + 3) % 4,) + prog[k + 1:]), r_cost))
        for nxt, c in steps:
            if nxt not in best or cost + c < best[nxt]:
                best[nxt] = cost + c
                heapq.heappush(heap, (cost + c, nxt))
    raise RuntimeError('no plan found')


def model_cost(cm, moves):
    """Cost (ms) of a robot moves string, with the cost constants of the Cubotino_moves module cm."""
    total = 0
    last = None
    for i in range(0, len(moves), 2):
        a, d = moves[i], int(moves[i + 1])
        if a == 'R':
            total += cm.ROTATE_AFTER_FLIP_COST if last == 'F' else cm.ROTATE_COST
        else:
            total += {'F': cm.FLIP_COST * d, 'S': cm.SPIN_COST}[a]
        last = a
    return total
