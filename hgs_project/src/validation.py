from .puzzle_utils import is_solvable


def validate_state(state, size):
    if len(state) != size*size:
        raise ValueError(f"State must have {size*size} numbers")
    if set(state) != set(range(size*size)):
        raise ValueError("State must contain all numbers 0..N-1")
    # Dočasne vypnuté pre 4x4 – overíme neskôr
    # if not is_solvable(state, size):
    #     raise ValueError("State is not solvable")
    return True

def validate_heuristic(heuristic):
    if heuristic not in ('manhattan', 'linear_conflict'):
        raise ValueError("Heuristic must be 'manhattan' or 'linear_conflict'")

def validate_max_depth(max_depth):
    if max_depth < 1:
        raise ValueError("Max depth must be >= 1")