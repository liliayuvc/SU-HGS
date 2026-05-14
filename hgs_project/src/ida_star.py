from .heuristics import Heuristic
from .puzzle_utils import get_successors

class IDASolver:
    def __init__(self, start, goal, size, heuristic='manhattan'):
        self.start = start
        self.goal = goal
        self.size = size
        self.heuristic = heuristic
        self.h_func = Heuristic.manhattan if heuristic == 'manhattan' else Heuristic.linear_conflict
        self.expanded_nodes = 0
        self.solution = []
        self.time_taken = 0.0

    def solve(self, max_depth=100):
        import time
        start_time = time.time()
        bound = self.h_func(self.start, self.goal, self.size)
        path = [self.start]
        while True:
            t = self._search(path, 0, bound)
            if t == 'FOUND':
                self.time_taken = time.time() - start_time
                return True
            if t == float('inf') or t > max_depth:
                self.time_taken = time.time() - start_time
                return False
            bound = t

    def _search(self, path, g, bound):
        node = path[-1]
        f = g + self.h_func(node, self.goal, self.size)
        if f > bound:
            return f
        if node == self.goal:
            self.solution = path[:]
            return 'FOUND'
        next_bound = float('inf')
        for succ in get_successors(node, self.size):
            if succ not in path:   # avoid cycles
                self.expanded_nodes += 1
                path.append(succ)
                t = self._search(path, g+1, bound)
                if t == 'FOUND':
                    return 'FOUND'
                if t < next_bound:
                    next_bound = t
                path.pop()
        return next_bound