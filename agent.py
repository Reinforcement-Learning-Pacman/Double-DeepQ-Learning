import random
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as f

from model import DQNModel
from replay_buffer import ReplayBuffer


class DoubleDQNAgent:
    def __init__(
            self,
            state_shape: Tuple[int, ...],
            n_actions: int,
            device: str,
            learning_rate: float,
            gamma: float,
            epsilon_start: float,
            epsilon_final: float,
            epsilon_decay: int,
            buffer_size: int,
            batch_size: int,
            target_update: int
    ) -> None:

        self.state_shape = state_shape
        self.n_actions = n_actions
        self.device = device

        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        # Networks
        self.policy_net = DQNModel(state_shape, n_actions).to(device)
        self.target_net = DQNModel(state_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.buffer = ReplayBuffer(buffer_size)
        self.steps_done = 0

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training:
            self.epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                           np.exp(-self.steps_done / self.epsilon_decay)
            self.steps_done += 1
            if random.random() < self.epsilon:
                return random.randrange(self.n_actions)

        with torch.no_grad(): # skip caculate gradient
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.max(1)[1].item()

    def optimize(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1) # chọn actions bằng policy network

        # Bellman formula: y = r + y * Q(s', a', θ')
        # self.gamma là hệ số discount (γ)
        next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1) # Tính giá trị Q cho next actions bằng target network
        next_q_values = next_q_values * (1 - dones) # if final state -> Q=0
        expected_q_values = rewards + self.gamma * next_q_values

        # Loss L = (y - Q(s,a;θ))²
        loss = f.smooth_l1_loss(q_values, expected_q_values)

        self.optimizer.zero_grad() # delete gradients from the last step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10) # limit gradient to ignore grandient exploding
        self.optimizer.step()

        return loss.item()

    def update_target_network(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict()) # copy weights from policy network to target network

    def save(self, path: str) -> None:
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
