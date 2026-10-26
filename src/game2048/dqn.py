from collections import deque, namedtuple
import random

import numpy as np
import torch
from torch import nn

Transition = namedtuple("Transition", "state action reward next_state done")


class QNetwork(nn.Module):
    def __init__(self, action_count=4):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 2 * 2, 128), nn.ReLU(),
            nn.Linear(128, action_count),
        )

    def forward(self, states):
        return self.layers(states)


class ReplayBuffer:
    def __init__(self, capacity):
        self.items = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.items.append(Transition(np.array(state, copy=True), action, reward,
                                     np.array(next_state, copy=True), done))

    def sample(self, batch_size, device):
        batch = random.sample(self.items, batch_size)
        states = torch.as_tensor(np.stack([item.state for item in batch]), dtype=torch.float32, device=device).unsqueeze(1)
        actions = torch.as_tensor([item.action for item in batch], dtype=torch.int64, device=device)
        rewards = torch.as_tensor([item.reward for item in batch], dtype=torch.float32, device=device)
        next_states = torch.as_tensor(np.stack([item.next_state for item in batch]), dtype=torch.float32, device=device).unsqueeze(1)
        dones = torch.as_tensor([item.done for item in batch], dtype=torch.float32, device=device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.items)


class DoubleDQNAgent:
    def __init__(self, learning_rate=1e-3, gamma=0.99, batch_size=128,
                 buffer_capacity=50000, target_update_interval=500, device="cpu"):
        self.device = torch.device(device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.online = QNetwork().to(self.device)
        self.target = QNetwork().to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.replay = ReplayBuffer(buffer_capacity)
        self.updates = 0

    def act(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return random.randrange(4)
        with torch.no_grad():
            tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
            return int(self.online(tensor).argmax(dim=1).item())

    def remember(self, *transition):
        self.replay.add(*transition)

    def learn(self):
        if len(self.replay) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size, self.device)
        current = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
            next_values = self.target(next_states).gather(1, next_actions).squeeze(1)
            expected = rewards + self.gamma * next_values * (1.0 - dones)
        loss = nn.functional.smooth_l1_loss(current, expected)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def save(self, path, episode, metrics):
        torch.save({"model": self.online.state_dict(), "optimizer": self.optimizer.state_dict(),
                    "episode": episode, "metrics": metrics}, path)

    def load(self, path):
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        self.online.load_state_dict(checkpoint["model"])
        self.target.load_state_dict(checkpoint["model"])
        return checkpoint
