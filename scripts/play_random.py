import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from game2048 import ThreeByThree2048Env

env = ThreeByThree2048Env(render_mode="ansi")
for episode in range(3):
    _, info = env.reset(seed=episode)
    done = False
    while not done:
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated
    print(f"episode={episode + 1} {info}\n{env.render()}\n")
