import torch
import torch.nn.functional as f
import numpy as np
import random
from typing import Tuple, Optional

from model import DQNModel
from replay_buffer import ReplayBuffer


class DoubleDQNAgent:
    """Agent sử dụng Double Deep Q-Network"""

    def __init__(
            self,
            state_shape: Tuple[int, ...],
            n_actions: int,
            device: str,
            learning_rate: float = 1e-4,  
            gamma: float = 0.99,
            epsilon_start: float = 1.0,
            epsilon_final: float = 0.01,  
            epsilon_decay: int = 100000,  
            buffer_size: int = 100000,    
            batch_size: int = 32,
            target_update: int = 10000
    ) -> None:
        """Khởi tạo Double DQN Agent"""
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
        self.target_net.eval()  # Target network luôn ở chế độ evaluation

        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)

        # Counter
        self.steps_done = 0

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Chọn action dựa trên chính sách epsilon-greedy"""
        if training:
            # Cập nhật epsilon
            self.epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                           np.exp(-self.steps_done / self.epsilon_decay)
            self.steps_done += 1

            # Exploration
            if random.random() < self.epsilon:
                return random.randrange(self.n_actions)

        # Exploitation
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.max(1)[1].item()

    def optimize(self) -> Optional[float]:
        """Huấn luyện mạng neural với một batch từ replay buffer"""
        # Kiểm tra xem buffer có đủ dữ liệu không
        if len(self.buffer) < self.batch_size:
            return None

        # Lấy batch từ replay buffer
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Chuyển sang tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Tính current Q values
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: chọn actions bằng policy network
        next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)

        # Tính giá trị Q cho next actions bằng target network
        next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)

        # Mask cho các states terminal
        next_q_values = next_q_values * (1 - dones)

        # Tính expected Q values
        expected_q_values = rewards + self.gamma * next_q_values

        # Tính loss
        loss = f.smooth_l1_loss(q_values, expected_q_values)

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping để tránh exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self) -> None:
        """Cập nhật target network với trọng số từ policy network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str) -> None:
        """Lưu model"""
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, path)

    def load(self, path: str) -> None:
        """Tải model"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']