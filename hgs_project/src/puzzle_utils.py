import random
from collections import deque

def is_solvable(state, size):
    """Returns True if the puzzle state is solvable."""
    inv_count = 0
    flat = [x for x in state if x != 0]
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            if flat[i] > flat[j]:
                inv_count += 1
 
    if size % 2 == 1:
        return inv_count % 2 == 0
    else:
        
        blank_row_from_bottom = size - (state.index(0) // size)
        if blank_row_from_bottom % 2 == 0:
            
            return inv_count % 2 == 1
        else:
           
            return inv_count % 2 == 0


def get_successors(state, size):
    zero = state.index(0)
    row, col = divmod(zero, size)
    moves = []
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = row+dr, col+dc
        if 0 <= nr < size and 0 <= nc < size:
            new_zero = nr*size + nc
            lst = list(state)
            lst[zero], lst[new_zero] = lst[new_zero], lst[zero]
            moves.append(tuple(lst))
    return moves

def format_state(state, size):
    return "\n".join(" ".join(str(state[i*size+j]) for j in range(size)) for i in range(size))

def random_state(size, steps=100):
    """Generate random solvable state by random moves from goal."""
    goal = tuple(range(size*size))
    state = goal
    for _ in range(steps):
        succ = get_successors(state, size)
        if succ:
            state = random.choice(succ)
    return state

def generate_state_with_min_depth(size, min_depth, max_attempts=500):
    """BFS from goal until depth >= min_depth (guaranteed solvable)."""
    goal = tuple(range(size*size))
    visited = {goal: 0}
    queue = deque([goal])
    best_state = goal
    best_depth = 0
    attempts = 0
    while queue and attempts < max_attempts:
        state = queue.popleft()
        depth = visited[state]
        if depth >= min_depth:
            return state
        if depth > best_depth:
            best_depth = depth
            best_state = state
        for succ in get_successors(state, size):
            if succ not in visited:
                visited[succ] = depth + 1
                queue.append(succ)
        attempts += 1
    return best_state
    
def evaluate(y_true, y_pred):
    """
    Compute classification metrics.
    Returns dict with: accuracy, precision, recall, f1, tp, tn, fp, fn
    """
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
 
    total = len(y_true)
    accuracy  = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp)    if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn)    if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
 
    return {
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }
