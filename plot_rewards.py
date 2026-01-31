import argparse
import csv

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default="artifacts/reward.csv")
    args = parser.parse_args()

    steps = []
    rewards = []
    with open(args.log, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["steps"]))
            rewards.append(float(row["avg_return"]))

    if not steps:
        print("no data found")
        return

    plt.figure(figsize=(6, 4))
    plt.plot(steps, rewards, color="#ee6c4d")
    plt.xlabel("steps")
    plt.ylabel("avg_return")
    plt.title("training curve")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
