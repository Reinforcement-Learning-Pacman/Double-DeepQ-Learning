import numpy as np
import random
from collections import deque
from typing import Tuple


class ReplayBuffer:
    """Bộ nhớ đệm để lưu trữ và lấy mẫu kinh nghiệm cho DQN"""

    def __init__(self, capacity: int) -> None:
        """Khởi tạo replay buffer"""
        self.buffer = deque(maxlen=capacity)

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Thêm một transition vào buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Lấy ngẫu nhiên một batch từ buffer"""
        # Kiểm tra kích thước buffer
        assert len(self.buffer) >= batch_size, "Buffer không đủ dữ liệu để lấy mẫu"

        # Lấy mẫu ngẫu nhiên
        batch = random.sample(self.buffer, batch_size)

        # Tách batch thành các thành phần
        states, actions, rewards, next_states, dones = zip(*batch)

        # Chuyển đổi sang numpy arrays
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.uint8)
        )

    def __len__(self) -> int:
        """Lấy kích thước hiện tại của buffer"""
        return len(self.buffer)