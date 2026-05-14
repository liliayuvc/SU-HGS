import argparse
from .ida_star import IDASolver
from .puzzle_utils import random_state, generate_state_with_min_depth, format_state
from .validation import validate_state, validate_heuristic, validate_max_depth

def main():
    parser = argparse.ArgumentParser(description="IDA* solver for N-puzzle (HGS)")
    parser.add_argument('--size', type=int, default=3, choices=[3,4])
    parser.add_argument('--heuristic', default='manhattan', choices=['manhattan','linear_conflict'])
    parser.add_argument('--start', nargs='+', type=int, help="custom start state")
    parser.add_argument('--max-depth', type=int, default=100)
    parser.add_argument('--min-depth', type=int, help="generate state with min solution depth")
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    goal = tuple(range(args.size*args.size))
    if args.start:
        start = tuple(args.start)
    elif args.min_depth:
        start = generate_state_with_min_depth(args.size, args.min_depth)
        print(f"Generated state with solution depth >= {args.min_depth}")
    else:
        start = random_state(args.size, steps=100)

    try:
        validate_state(start, args.size)
        validate_heuristic(args.heuristic)
        validate_max_depth(args.max_depth)
    except ValueError as e:
        print(f"Validation error: {e}")
        return

    solver = IDASolver(start, goal, args.size, args.heuristic)
    print(f"Solving {args.size}x{args.size} puzzle with {args.heuristic} heuristic...")
    ok = solver.solve(max_depth=args.max_depth)

    if ok:
        print(f" Solution found in {len(solver.solution)-1} moves")
        print(f" Expanded nodes: {solver.expanded_nodes}")
        print(f" Time: {solver.time_taken:.3f} seconds")
        if args.verbose:
            for i, state in enumerate(solver.solution):
                print(f"\nStep {i}:")
                print(format_state(state, args.size))
    else:
        print(" No solution found (max depth exceeded or unsolvable)")

if __name__ == '__main__':
    main()