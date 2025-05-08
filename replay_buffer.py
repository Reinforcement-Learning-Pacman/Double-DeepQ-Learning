import random
from collections import deque
from typing import Tuple, List
import numpy as np
import torch


class SumTree:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity  
        self.tree = np.zeros(2 * capacity - 1)  
        self.data = np.zeros(capacity, dtype=object)  
        self.n_entries = 0  
        self.write = 0  

    def _propagate(self, idx: int, change: float) -> None:
        """Lan truyền thay đổi lên các node cha"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Tìm chỉ mục của lá dựa trên giá trị s"""
        left = 2 * idx + 1
        right = left + 1

        
        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """Trả về tổng ưu tiên"""
        return self.tree[0]

    def add(self, priority: float, data) -> None:
        """Thêm dữ liệu và ưu tiên của nó vào cây"""
        idx = self.write + self.capacity - 1  
        
        self.data[self.write] = data
        self.update(idx, priority)

        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx: int, priority: float) -> None:
        """Cập nhật ưu tiên"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, object]:
        """Lấy dữ liệu dựa trên giá trị s"""
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[dataIdx]


class PrioritizedReplayBuffer:
    """Replay buffer với ưu tiên theo TD error"""
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001, epsilon: float = 0.01) -> None:
        """
        Args:
            capacity: Kích thước buffer
            alpha: Mức độ ưu tiên (alpha=0 là sampling đều, alpha=1 là sampling hoàn toàn theo ưu tiên)
            beta: Mức độ điều chỉnh importance sampling (beta=1 là không thiên lệch hoàn toàn)
            beta_increment: Tốc độ tăng beta trong quá trình học (cuối cùng beta sẽ đạt 1)
            epsilon: Giá trị nhỏ để đảm bảo mọi trải nghiệm đều có cơ hội được lấy mẫu
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0  # Ưu tiên tối đa ban đầu
    
    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Thêm transition vào buffer với ưu tiên tối đa"""
        transition = (state, action, reward, next_state, done)
        
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, transition)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], np.ndarray]:
        """Lấy mẫu batch theo ưu tiên"""
        batch = []
        indices = []
        priorities = []
        
        
        segment = self.tree.total() / batch_size
        
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        for i in range(batch_size):
            
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            
            idx, priority, data = self.tree.get(s)
            priorities.append(priority)
            batch.append(data)
            indices.append(idx)

       
        sampling_probabilities = np.array(priorities) / self.tree.total()
        weights = (self.tree.n_entries * sampling_probabilities) ** (-self.beta)
        weights = weights / weights.max()  
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.uint8),
            indices,
            np.array(weights, dtype=np.float32)
        )
    
    def update_priorities(self, indices: List[int], td_errors: np.ndarray) -> None:
        """Cập nhật ưu tiên dựa trên TD errors"""
        for idx, td_error in zip(indices, td_errors):
            
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            
            
            self.max_priority = max(self.max_priority, priority)
            
            
            self.tree.update(idx, priority)
    
    def __len__(self) -> int:
        return self.tree.n_entries