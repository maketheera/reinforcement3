import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from game2048 import ThreeByThree2048Env
from game2048.dqn import DoubleDQNAgent


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained 3x3 2048 agent")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    env = ThreeByThree2048Env(max_steps=args.max_steps, render_mode="ansi")
    agent = DoubleDQNAgent(device=device)
    agent.load(args.checkpoint)
    outcomes = []
    for episode in range(args.episodes):
        state, _ = env.reset(seed=args.seed + episode)
        while True:
            action = agent.act(state)
            state, _, terminated, truncated, info = env.step(action)
            if args.render:
                print(f"episode={episode + 1} step={info['steps']}\n{env.render()}\n")
            if terminated or truncated:
                break
        outcomes.append(info)
    successful_sums = [item["board_sum"] for item in outcomes if item["success"]]
    print(f"episodes={args.episodes}")
    print(f"success_rate={np.mean([item['success'] for item in outcomes]):.3f}")
    print(f"avg_board_sum={np.mean([item['board_sum'] for item in outcomes]):.2f}")
    print(f"avg_success_board_sum={np.mean(successful_sums) if successful_sums else 0.0:.2f}")
    print(f"avg_max_tile={np.mean([item['max_tile'] for item in outcomes]):.2f}")
    print(f"avg_steps={np.mean([item['steps'] for item in outcomes]):.2f}")


if __name__ == "__main__":
    main()
