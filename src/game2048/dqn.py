from collections import deque, namedtuple
import random

import numpy as np
import torch
from torch import nn

Transition = namedtuple("Transition", "state action reward next_state done next_mask")


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

    def add(self, state, action, reward, next_state, done, next_mask):
        self.items.append(Transition(np.array(state, copy=True), action, reward,
                                     np.array(next_state, copy=True), done,
                                     np.array(next_mask, dtype=np.bool_, copy=True)))

    def sample(self, batch_size, device):
        batch = random.sample(self.items, batch_size)
        states = torch.as_tensor(np.stack([item.state for item in batch]), dtype=torch.float32, device=device).unsqueeze(1)
        actions = torch.as_tensor([item.action for item in batch], dtype=torch.int64, device=device)
        rewards = torch.as_tensor([item.reward for item in batch], dtype=torch.float32, device=device)
        next_states = torch.as_tensor(np.stack([item.next_state for item in batch]), dtype=torch.float32, device=device).unsqueeze(1)
        dones = torch.as_tensor([item.done for item in batch], dtype=torch.float32, device=device)
        next_masks = torch.as_tensor(np.stack([item.next_mask for item in batch]), dtype=torch.bool, device=device)
        return states, actions, rewards, next_states, dones, next_masks

    def __len__(self):
        return len(self.items)


class DoubleDQNAgent:
    def __init__(self, learning_rate=5e-4, gamma=0.99, batch_size=128,
                 buffer_capacity=100000, target_update_interval=1000, device="cpu"):
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

    def act(self, state, epsilon=0.0, legal_mask=None):
        if legal_mask is None:
            legal_mask = np.ones(4, dtype=np.bool_)
        legal_indices = np.flatnonzero(legal_mask)
        if len(legal_indices) == 0:
            legal_indices = np.arange(4)
        if random.random() < epsilon:
            return int(random.choice(legal_indices.tolist()))
        with torch.no_grad():
            tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
            values = self.online(tensor).squeeze(0)
            mask = torch.as_tensor(legal_mask, dtype=torch.bool, device=self.device)
            values = values.masked_fill(~mask, torch.finfo(values.dtype).min)
            return int(values.argmax().item())

    def remember(self, state, action, reward, next_state, done, next_mask):
        self.replay.add(state, action, reward, next_state, done, next_mask)

    def learn(self):
        if len(self.replay) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones, next_masks = self.replay.sample(self.batch_size, self.device)
        current = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_online = self.online(next_states).masked_fill(~next_masks, torch.finfo(torch.float32).min)
            next_actions = next_online.argmax(dim=1, keepdim=True)
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
