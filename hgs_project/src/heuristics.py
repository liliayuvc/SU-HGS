class Heuristic:
    @staticmethod
    def manhattan(state, goal, size):
        dist = 0
        for i, val in enumerate(state):
            if val != 0:
                goal_i = goal.index(val)
                dr = abs(i//size - goal_i//size)
                dc = abs(i%size - goal_i%size)
                dist += dr + dc
        return dist

    @staticmethod
    def linear_conflict(state, goal, size):
        """Manhattan + linear conflicts (more specific)."""
        man = Heuristic.manhattan(state, goal, size)
        conflict = 0
        # rows
        for r in range(size):
            row_vals = [state[r*size + c] for c in range(size) if state[r*size + c] != 0]
            row_goal = [goal[r*size + c] for c in range(size) if goal[r*size + c] != 0]
            for a in range(len(row_vals)):
                for b in range(a+1, len(row_vals)):
                    if (row_vals[a] in row_goal and row_vals[b] in row_goal and
                        row_goal.index(row_vals[a]) > row_goal.index(row_vals[b])):
                        conflict += 2
        # columns
        for c in range(size):
            col_vals = [state[r*size + c] for r in range(size) if state[r*size + c] != 0]
            col_goal = [goal[r*size + c] for r in range(size) if goal[r*size + c] != 0]
            for a in range(len(col_vals)):
                for b in range(a+1, len(col_vals)):
                    if (col_vals[a] in col_goal and col_vals[b] in col_goal and
                        col_goal.index(col_vals[a]) > col_goal.index(col_vals[b])):
                        conflict += 2
        return man + conflict

from typing import List, Optional
import numpy as np
 
WILDCARD = "?"
NONE_SYM = "∅"
 
 
class Hypothesis:
    def __init__(self, slots: List[str]):
        self.slots = list(slots)
 
    @classmethod
    def most_general(cls, n_features: int) -> "Hypothesis":
        return cls([WILDCARD] * n_features)
 
    @classmethod
    def most_specific(cls, n_features: int) -> "Hypothesis":
        return cls([NONE_SYM] * n_features)
 
    def covers(self, example: List[str]) -> bool:
        for slot, val in zip(self.slots, example):
            if slot == NONE_SYM:
                return False
            if slot == WILDCARD:
                continue
            if slot != val:
                return False
        return True
 
    def is_more_general_than(self, other: "Hypothesis") -> bool:
        if self.slots == other.slots:
            return False
        for s, o in zip(self.slots, other.slots):
            if s != WILDCARD and s != o:
                return False
        return True
 
    def generalise_to(self, example: List[str]) -> "Hypothesis":
        new_slots = []
        for slot, val in zip(self.slots, example):
            if slot == NONE_SYM:
                new_slots.append(val)
            elif slot == WILDCARD or slot == val:
                new_slots.append(slot)
            else:
                new_slots.append(WILDCARD)
        return Hypothesis(new_slots)
 
    def specialise_to_exclude(self, example, possible_values):
        specialisations = []
        for i, (slot, val) in enumerate(zip(self.slots, example)):
            if slot == WILDCARD:
                for v in possible_values[i]:
                    if v != val:
                        new_slots = list(self.slots)
                        new_slots[i] = v
                        specialisations.append(Hypothesis(new_slots))
        return specialisations
 
    def copy(self):
        return Hypothesis(list(self.slots))
 
    def __eq__(self, other):
        return isinstance(other, Hypothesis) and self.slots == other.slots
 
    def __hash__(self):
        return hash(tuple(self.slots))
 
    def __repr__(self):
        return f"<{', '.join(self.slots)}>"
 
    def __str__(self):
        return self.__repr__()
 
 
class HGS:
    def __init__(self, feature_names=None):
        self.feature_names = feature_names
        self.g_set = []
        self.s_set = []
        self.history_ = []
        self.is_version_space_empty = False
        self._possible_values = []
 
    def fit(self, X, y):
        n_features = len(X[0])
        self._possible_values = [list({row[i] for row in X}) for i in range(n_features)]
        self.g_set = [Hypothesis.most_general(n_features)]
        self.s_set = [Hypothesis.most_specific(n_features)]
 
        for xi, yi in zip(X, y):
            if yi == 1:
                self._process_positive(xi)
            else:
                self._process_negative(xi)
            self.history_.append({"g_size": len(self.g_set), "s_size": len(self.s_set)})
            if not self.g_set or not self.s_set:
                self.is_version_space_empty = True
        return self
 
    def _process_positive(self, xi):
        self.g_set = [h for h in self.g_set if h.covers(xi)]
        new_s = []
        for h in self.s_set:
            if not h.covers(xi):
                gen = h.generalise_to(xi)
                if any(g.is_more_general_than(gen) or g == gen for g in self.g_set):
                    new_s.append(gen)
            else:
                new_s.append(h)
        self.s_set = self._remove_more_general(new_s)
 
    def _process_negative(self, xi):
        self.s_set = [h for h in self.s_set if not h.covers(xi)]
        new_g = []
        for h in self.g_set:
            if h.covers(xi):
                specs = h.specialise_to_exclude(xi, self._possible_values)
                for s in specs:
                    if any(s.is_more_general_than(sh) or s == sh for sh in self.s_set):
                        if not any(g.is_more_general_than(s) for g in new_g):
                            new_g.append(s)
            else:
                new_g.append(h)
        self.g_set = self._remove_more_specific(new_g)
 
    def _remove_more_general(self, hypotheses):
        result = []
        for h in hypotheses:
            if not any(o != h and o.is_more_general_than(h) for o in hypotheses):
                if h not in result:
                    result.append(h)
        return result
 
    def _remove_more_specific(self, hypotheses):
        result = []
        for h in hypotheses:
            if not any(o != h and h.is_more_general_than(o) for o in hypotheses):
                if h not in result:
                    result.append(h)
        return result
 
    def predict(self, X):
        return [1 if any(h.covers(xi) for h in self.g_set) else 0 for xi in X]
 
    def predict_proba(self, X):
        preds = self.predict(X)
        result = np.zeros((len(X), 2))
        for i, p in enumerate(preds):
            result[i][p] = 1.0
        return result
 
    def summary(self):
        lines = [
            "HGS Summary",
            "-" * 30,
            f"G-set size : {len(self.g_set)}",
            f"S-set size : {len(self.s_set)}",
            f"Version space empty: {self.is_version_space_empty}",
            "",
            "G-set (most general):",
        ]
        for h in self.g_set:
            lines.append(f"  {h}")
        lines.append("S-set (most specific):")
        for h in self.s_set:
            lines.append(f"  {h}")
        return "\n".join(lines)

