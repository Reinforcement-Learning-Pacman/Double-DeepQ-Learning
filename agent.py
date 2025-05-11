import random  # Thư viện để sinh số ngẫu nhiên, sử dụng trong chiến lược epsilon-greedy.
from typing import Tuple, Optional  # Type hinting để làm rõ kiểu dữ liệu của các tham số và giá trị trả về.

import numpy as np  # NumPy cho các phép toán mảng, đặc biệt là xử lý state và epsilon decay.
import torch  # PyTorch, thư viện chính cho deep learning.
import torch.nn.functional as f # Module functional của PyTorch, thường được alias là F, nhưng ở đây dùng f.
                                # Chứa các hàm như loss functions, activation functions.

from model import DQNModel  # Import kiến trúc mạng DQN từ file model.py (giả định).
from replay_buffer import PrioritizedReplayBuffer  # Import lớp Prioritized Replay Buffer.


class DoubleDQNAgent:
    """
    Lớp DoubleDQNAgent triển khai thuật toán Double Deep Q-Network với Prioritized Experience Replay.
    """
    def __init__(  
            self,
            state_shape: Tuple[int, ...],  # Hình dạng của state đầu vào (ví dụ: (channels, height, width)).
            n_actions: int,  # Số lượng hành động mà agent có thể thực hiện.
            device: str,  # Thiết bị để chạy tính toán ('cpu' hoặc 'cuda').
            learning_rate: float = 5e-5,  # Tốc độ học cho optimizer.
            gamma: float = 0.99,  # Hệ số chiết khấu (discount factor) cho phần thưởng tương lai.
            epsilon_start: float = 1.0,  # Giá trị ban đầu của epsilon (cho exploration).
            epsilon_final: float = 0.01,  # Giá trị cuối cùng của epsilon.
            epsilon_decay: int = 220000,  # Số bước để epsilon giảm từ start xuống final.
            buffer_size: int = 100000,    # Kích thước tối đa của replay buffer.
            batch_size: int = 32,  # Số lượng mẫu lấy từ buffer cho mỗi lần cập nhật mạng.
            target_update: int = 10000  # Tần suất (theo số bước) cập nhật target network.
    ) -> None:

        self.state_shape = state_shape # Lưu trữ hình dạng state.
        self.n_actions = n_actions     # Lưu trữ số lượng hành động.
        self.device = device           # Lưu trữ thiết bị tính toán.

        self.gamma = gamma              # Lưu trữ hệ số chiết khấu.
        self.epsilon = epsilon_start    # Khởi tạo epsilon hiện tại bằng giá trị bắt đầu.
        self.epsilon_start = epsilon_start # Lưu trữ epsilon bắt đầu.
        self.epsilon_final = epsilon_final # Lưu trữ epsilon cuối cùng.
        self.epsilon_decay = epsilon_decay # Lưu trữ tốc độ giảm epsilon.
        self.batch_size = batch_size    # Lưu trữ kích thước batch.
        self.target_update = target_update # Lưu trữ tần suất cập nhật target network.

        # --- Khởi tạo Mạng Neural ---
        # Policy Network (mạng chính): Mạng này được huấn luyện liên tục.
        # Nó được sử dụng để chọn hành động (trong quá trình exploitation) và để đánh giá Q-value của hành động được chọn bởi chính nó (trong DQN chuẩn).
        self.policy_net = DQNModel(state_shape, n_actions).to(device)
        # Target Network: Mạng này có kiến trúc giống hệt policy_net.
        # Trọng số của nó được cập nhật định kỳ bằng cách sao chép từ policy_net.
        # Nó được sử dụng để cung cấp các giá trị Q mục tiêu (target Q-values) ổn định hơn, giúp giảm dao động trong quá trình huấn luyện.
        self.target_net = DQNModel(state_shape, n_actions).to(device)
        # Sao chép trọng số từ policy_net sang target_net khi khởi tạo.
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # Chuyển target_net sang chế độ đánh giá (evaluation mode).
        # Điều này quan trọng vì nó sẽ tắt các cơ chế như Dropout và Batch Normalization (nếu có) không cần thiết khi tính toán target Q-values.
        self.target_net.eval()

        # Optimizer: Sử dụng Adam optimizer để cập nhật các tham số của policy_net.
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        # Replay Buffer: Sử dụng Prioritized Replay Buffer.
        # Nó lưu trữ các kinh nghiệm (state, action, reward, next_state, done) và
        # ưu tiên lấy mẫu các kinh nghiệm "quan trọng" hơn (có TD error lớn hơn).
        self.buffer = PrioritizedReplayBuffer(
            capacity=buffer_size, # Kích thước tối đa của buffer.
            alpha=0.6,  # Tham số alpha: Kiểm soát mức độ ưu tiên (0: uniform, 1: full priority).
            beta=0.4,   # Tham số beta: Bù đắp cho sự thiên lệch do lấy mẫu ưu tiên (Importance Sampling).
                        # Beta sẽ tăng dần từ giá trị này lên 1.0.
            beta_increment=0.0001  # Lượng tăng beta sau mỗi lần lấy mẫu.
        )
        self.steps_done = 0  # Biến đếm tổng số bước (hành động) đã thực hiện, dùng cho epsilon decay.

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Chọn hành động dựa trên state hiện tại theo chiến lược epsilon-greedy (nếu training).
        Nếu không training, chọn hành động tốt nhất (exploitation).
        """
        if training:
            # Tính toán giá trị epsilon hiện tại dựa trên công thức suy giảm theo hàm mũ.
            # Epsilon sẽ giảm dần từ epsilon_start xuống epsilon_final qua `epsilon_decay` bước.
            self.epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                           np.exp(-self.steps_done / self.epsilon_decay)
            self.steps_done += 1 # Tăng biến đếm số bước đã thực hiện.

            # Exploration: Với xác suất epsilon, chọn một hành động ngẫu nhiên.
            if random.random() < self.epsilon:
                return random.randrange(self.n_actions) # Trả về một hành động ngẫu nhiên từ 0 đến n_actions-1.

        # Exploitation: Với xác suất (1-epsilon) hoặc nếu không training, chọn hành động tốt nhất.
        with torch.no_grad(): # Tắt tính toán gradient vì đây là quá trình inference, không cần backprop.
            # Chuyển state (NumPy array) thành PyTorch FloatTensor.
            # `unsqueeze(0)` thêm một chiều batch (ví dụ: từ (C, H, W) thành (1, C, H, W)).
            # `.to(self.device)` chuyển tensor lên thiết bị tính toán (CPU/GPU).
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            # Đưa state_tensor qua policy_net để nhận Q-values cho tất cả các hành động.
            q_values = self.policy_net(state_tensor)
            # Chọn hành động có Q-value lớn nhất.
            # `q_values.max(1)` trả về (giá trị lớn nhất, chỉ số của giá trị lớn nhất) dọc theo chiều 1 (chiều của các hành động).
            # `[1]` lấy chỉ số (tức là hành động).
            # `.item()` chuyển tensor một phần tử thành số Python.
            return q_values.max(1)[1].item()

    def optimize(self) -> Optional[float]:
        """
        Thực hiện một bước tối ưu hóa mạng neural bằng cách lấy mẫu từ replay buffer.
        Trả về giá trị loss nếu tối ưu hóa được thực hiện, ngược lại trả về None.
        """
        # Chỉ bắt đầu tối ưu hóa khi buffer có đủ số mẫu cho một batch.
        if len(self.buffer) < self.batch_size:
            return None # Chưa đủ mẫu, không làm gì cả.

        # Lấy một batch các kinh nghiệm từ Prioritized Replay Buffer.
        # Bao gồm: states, actions, rewards, next_states, dones, indices (cho việc cập nhật ưu tiên),
        # và weights (cho importance sampling).
        states, actions, rewards, next_states, dones, indices, weights = self.buffer.sample(self.batch_size)

        # Chuyển đổi các mảng NumPy từ buffer thành PyTorch tensors và chuyển lên device.
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device) # Hành động là chỉ số, nên dùng LongTensor.
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device) # Cờ done (0 hoặc 1).
        weights = torch.FloatTensor(weights).to(self.device) # Importance sampling weights.

        # --- Tính toán Q-values hiện tại (current Q-values) ---
        # Đưa `states` qua `policy_net` để lấy Q-values cho tất cả hành động.
        # `gather(1, actions.unsqueeze(1))` chọn ra Q-value của hành động thực sự đã thực hiện (`actions`).
        # `actions.unsqueeze(1)` thay đổi shape của actions từ (batch_size,) thành (batch_size, 1).
        # `squeeze(1)` loại bỏ chiều không cần thiết, kết quả là (batch_size,).
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # --- Tính toán Q-values mục tiêu (target Q-values) theo cơ chế Double DQN ---
        # 1. Chọn hành động tốt nhất cho `next_states` bằng `policy_net`.
        #    `self.policy_net(next_states)`: (batch_size, n_actions)
        #    `.max(1)[1]`: lấy chỉ số của hành động có Q-value lớn nhất, shape (batch_size,)
        #    `.unsqueeze(1)`: shape (batch_size, 1) -> để dùng với gather.
        next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)

        # 2. Đánh giá Q-value của các `next_actions` đó bằng `target_net`.
        #    Điều này giúp giảm overestimation bias so với DQN chuẩn.
        #    `self.target_net(next_states)`: (batch_size, n_actions)
        #    `.gather(1, next_actions)`: chọn Q-value của hành động `next_actions`.
        #    `.squeeze(1)`: shape (batch_size,).
        next_q_values_target_net = self.target_net(next_states).gather(1, next_actions).squeeze(1)

        # Nếu state tiếp theo là terminal (done=1), thì Q-value của nó là 0.
        # (1 - dones) sẽ là 0 nếu done=1 (terminal), và 1 nếu done=0 (non-terminal).
        next_q_values_target_net = next_q_values_target_net * (1 - dones)

        # Tính toán Q-values mục tiêu kỳ vọng theo công thức Bellman: R + gamma * Q_target(s', a').
        expected_q_values = rewards + self.gamma * next_q_values_target_net

        # --- Tính toán TD errors để cập nhật ưu tiên trong buffer ---
        # TD error = Q_current - Q_expected.
        # `.detach()`: tách tensor này ra khỏi đồ thị tính toán, vì TD error chỉ dùng để cập nhật buffer,
        # không dùng để tính gradient cho loss của mạng chính (loss đã được tính riêng).
        # `.cpu().numpy()`: chuyển về CPU và thành NumPy array để buffer xử lý.
        td_errors = torch.abs(q_values - expected_q_values).detach().cpu().numpy()

        # --- Tính toán Loss ---
        # Sử dụng Smooth L1 Loss (Huber loss).
        # `reduction='none'` để tính loss cho từng phần tử trong batch.
        # Nhân loss của mỗi mẫu với importance sampling weight `weights` tương ứng.
        # `.mean()`: tính trung bình loss của cả batch.
        loss = (weights * torch.nn.functional.smooth_l1_loss(q_values, expected_q_values, reduction='none')).mean()

        # --- Thực hiện bước tối ưu hóa ---
        self.optimizer.zero_grad()  # Xóa các gradient cũ.
        loss.backward()  # Tính toán gradient của loss theo các tham số của policy_net.
        # Clip gradient để tránh exploding gradients, giúp ổn định quá trình huấn luyện.
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)
        self.optimizer.step()  # Cập nhật các tham số của policy_net.

        # Cập nhật ưu tiên của các mẫu đã lấy trong replay buffer dựa trên TD errors mới tính được.
        self.buffer.update_priorities(indices, td_errors)

        return loss.item() # Trả về giá trị loss (dưới dạng số Python) để theo dõi.

    def update_target_network(self) -> None:
        """Sao chép trọng số từ policy_net sang target_net."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str) -> None:
        """
        Lưu trạng thái của agent (trọng số mạng, trạng thái optimizer, epsilon, steps_done) vào file.
        """
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(), # Trọng số của policy network.
            'target_state_dict': self.target_net.state_dict(), # Trọng số của target network.
            'optimizer_state_dict': self.optimizer.state_dict(),# Trạng thái của optimizer (để có thể tiếp tục huấn luyện).
            'epsilon': self.epsilon,                           # Giá trị epsilon hiện tại.
            'steps_done': self.steps_done                      # Tổng số bước đã thực hiện.
        }, path) # Lưu vào đường dẫn `path`.

    def load(self, path: str) -> None:
        """
        Tải trạng thái của agent từ file.
        """
        # Tải checkpoint từ file. `map_location=self.device` đảm bảo tensor được load lên đúng thiết bị.
        # `weights_only=False` (mặc định) cho phép load cả optimizer state và các thông tin khác.
        # Nếu chỉ load trọng số model, có thể dùng `weights_only=True` cho an toàn hơn.
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']

        # Sau khi load, đảm bảo target_net vẫn ở chế độ evaluation.
        self.target_net.eval()
        # Và policy_net ở chế độ training nếu bạn định tiếp tục huấn luyện.
        # self.policy_net.train() # Bỏ comment nếu cần thiết, thường được xử lý ở vòng lặp huấn luyện chính.