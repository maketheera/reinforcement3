import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from game2048 import ThreeByThree2048Env
from game2048.dqn import DoubleDQNAgent


def main():
    parser = argparse.ArgumentParser(description="Train a Double DQN 3x3 2048 agent")
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    env = ThreeByThree2048Env(max_steps=args.max_steps)
    agent = DoubleDQNAgent(device=device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    decay_steps = max(1, int(args.episodes * 0.7))
    best_key = (-1.0, -1.0)
    for episode in range(1, args.episodes + 1):
        state, _ = env.reset(seed=args.seed + episode)
        epsilon = max(0.05, 1.0 - 0.95 * episode / decay_steps)
        losses = []
        while True:
            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.remember(state, action, reward, next_state, done)
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
            state = next_state
            if done:
                break
        key = (float(info["success"]), float(info["board_sum"]))
        if key > best_key:
            best_key = key
            agent.save(os.path.join(args.checkpoint_dir, "best.pt"), episode, info)
        if episode % 100 == 0 or episode == 1:
            print(f"episode={episode} epsilon={epsilon:.3f} success={info['success']} "
                  f"sum={info['board_sum']} max={info['max_tile']} steps={info['steps']} "
                  f"loss={np.mean(losses) if losses else 0.0:.4f}")
    print(f"Training complete. Best success/sum={best_key}; device={device}")


if __name__ == "__main__":
    main()
