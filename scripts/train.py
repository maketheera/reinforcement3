import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from game2048 import ThreeByThree2048Env
from game2048.dqn import DoubleDQNAgent


def resolve_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA requested but unavailable. torch.version.cuda={torch.version.cuda}; "
            "check the NVIDIA driver and install a compatible CUDA PyTorch build."
        )
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    return requested


def main():
    parser = argparse.ArgumentParser(description="Train a masked Double DQN 3x3 2048 agent")
    parser.add_argument("--episodes", type=int, default=50000)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    env = ThreeByThree2048Env(max_steps=args.max_steps)
    agent = DoubleDQNAgent(device=device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    decay_steps = max(1, int(args.episodes * 0.7))
    best_key = (-1.0, -1.0)
    successes = 0
    recent_successes = []
    for episode in range(1, args.episodes + 1):
        state, info = env.reset(seed=args.seed + episode)
        legal_mask = np.asarray(info["legal_actions"], dtype=np.bool_)
        epsilon = max(0.05, 1.0 - 0.95 * episode / decay_steps)
        losses = []
        while True:
            action = agent.act(state, epsilon, legal_mask)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_mask = np.asarray(info["legal_actions"], dtype=np.bool_)
            agent.remember(state, action, reward, next_state, done, next_mask)
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
            state, legal_mask = next_state, next_mask
            if done:
                break
        successes += int(info["success"])
        recent_successes.append(int(info["success"]))
        if len(recent_successes) > 100:
            recent_successes.pop(0)
        key = (float(info["success"]), float(info["board_sum"]))
        if key > best_key:
            best_key = key
            agent.save(os.path.join(args.checkpoint_dir, "best.pt"), episode, info)
        if episode % 100 == 0 or episode == 1:
            print(f"episode={episode} epsilon={epsilon:.3f} success={info['success']} "
                  f"success_rate_100={np.mean(recent_successes):.3f} sum={info['board_sum']} "
                  f"max={info['max_tile']} steps={info['steps']} "
                  f"loss={np.mean(losses) if losses else 0.0:.4f}")
    print(f"Training complete. success_rate={successes / args.episodes:.3f}; "
          f"best_success/sum={best_key}; device={device}")


if __name__ == "__main__":
    main()
