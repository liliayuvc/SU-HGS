import csv, random
from src.puzzle_utils import generate_state_with_min_depth, format_state

def generate_dataset(size, depths, samples_per_depth=5, filename="puzzles.csv"):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["depth_min", "state"])
        for depth in depths:
            for _ in range(samples_per_depth):
                state = generate_state_with_min_depth(size, depth)
                writer.writerow([depth, " ".join(map(str, state))])
    print(f"Dataset saved to {filename}")

if __name__ == "__main__":
    generate_dataset(3, depths=[5,10,15,20,25], samples_per_depth=10)
    generate_dataset(4, depths=[10,20,30,40], samples_per_depth=5)