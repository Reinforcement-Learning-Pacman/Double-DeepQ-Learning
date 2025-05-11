import random  # Để sinh số ngẫu nhiên, sử dụng trong việc lấy mẫu.
from collections import deque  # Không được sử dụng trực tiếp trong code này, nhưng thường hữu ích cho replay buffer tiêu chuẩn.
from typing import Tuple, List  # Type hinting để cải thiện khả năng đọc và phân tích tĩnh.
import numpy as np  # NumPy cho các phép toán mảng hiệu quả, đặc biệt cho SumTree và lưu trữ dữ liệu.
import torch # PyTorch, không được sử dụng trực tiếp trong file này nhưng thường được sử dụng cùng với replay buffer.


class SumTree:
    """
    Cấu trúc dữ liệu SumTree (Cây tổng).
    Đây là một cây nhị phân mà mỗi node lá lưu trữ "ưu tiên" của một mẫu dữ liệu,
    và mỗi node bên trong (internal node) lưu trữ tổng ưu tiên của các con của nó.
    Gốc của cây lưu trữ tổng ưu tiên của tất cả các mẫu.
    Cấu trúc này cho phép lấy mẫu hiệu quả theo phân phối ưu tiên (O(log N))
    và cập nhật ưu tiên cũng hiệu quả (O(log N)).
    """
    def __init__(self, capacity: int) -> None:
        """
        Khởi tạo SumTree.

        Args:
            capacity (int): Số lượng mẫu tối đa mà cây có thể lưu trữ (số lượng node lá).
        """
        self.capacity = capacity  # Số lượng node lá tối đa.
        # Cây được biểu diễn bằng một mảng NumPy.
        # Kích thước của mảng là 2 * capacity - 1.
        # Các node lá nằm ở nửa sau của mảng (từ chỉ mục `capacity - 1` đến `2 * capacity - 2`).
        # Các node bên trong nằm ở nửa đầu (từ chỉ mục `0` đến `capacity - 2`).
        self.tree = np.zeros(2 * capacity - 1)
        # Mảng để lưu trữ dữ liệu thực tế (ví dụ: các transition (s, a, r, s', done)).
        # `dtype=object` cho phép lưu trữ các đối tượng Python bất kỳ (như tuple transition).
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0  # Số lượng mẫu hiện đang được lưu trữ trong cây.
        self.write = 0  # Con trỏ đến vị trí tiếp theo để ghi dữ liệu vào mảng `self.data` (hoạt động như một buffer vòng).

    def _propagate(self, idx: int, change: float) -> None:
        """
        Lan truyền sự thay đổi ưu tiên từ một node lá lên các node cha của nó.
        Khi ưu tiên của một node lá thay đổi, tất cả các node cha của nó cũng phải được cập nhật.

        Args:
            idx (int): Chỉ mục của node trong mảng `self.tree` có giá trị thay đổi.
            change (float): Lượng thay đổi của ưu tiên.
        """
        parent = (idx - 1) // 2  # Tính chỉ mục của node cha.
        self.tree[parent] += change  # Cập nhật giá trị của node cha.
        if parent != 0:  # Nếu node cha không phải là gốc (chỉ mục 0),
            self._propagate(parent, change)  # tiếp tục lan truyền thay đổi lên trên.

    def _retrieve(self, idx: int, s: float) -> int:
        """
        Tìm chỉ mục của node lá dựa trên một giá trị `s` (một giá trị được lấy mẫu từ [0, total_priority]).
        Hàm này duyệt cây từ gốc xuống lá để tìm node lá tương ứng với `s`.

        Args:
            idx (int): Chỉ mục của node hiện tại đang được xem xét (bắt đầu từ gốc, idx=0).
            s (float): Giá trị được lấy mẫu.

        Returns:
            int: Chỉ mục của node lá được tìm thấy trong mảng `self.tree`.
        """ 
        left = 2 * idx + 1  # Chỉ mục của con trái.
        right = left + 1  # Chỉ mục của con phải.

        # Nếu node con trái vượt ra ngoài phạm vi của cây, điều này có nghĩa là
        # `idx` hiện tại là một node lá (hoặc cây trống và `idx` là gốc).
        if left >= len(self.tree):
            return idx

        # Nếu `s` nhỏ hơn hoặc bằng giá trị của node con trái,
        # đi xuống nhánh con trái.
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        # Ngược lại, đi xuống nhánh con phải.
        # Giá trị `s` được trừ đi giá trị của node con trái vì chúng ta đã "đi qua" nhánh đó.
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """Trả về tổng ưu tiên của tất cả các mẫu (giá trị của node gốc)."""
        return self.tree[0]

    def add(self, priority: float, data) -> None:
        """
        Thêm một mẫu dữ liệu mới và ưu tiên của nó vào cây.
        Nếu cây đã đầy, mẫu cũ nhất sẽ bị ghi đè.

        Args:
            priority (float): Ưu tiên của mẫu dữ liệu mới.
            data: Mẫu dữ liệu (ví dụ: một tuple transition).
        """
        # `idx` là chỉ mục của node lá trong mảng `self.tree` nơi ưu tiên sẽ được lưu trữ.
        # Nó tương ứng với vị trí `self.write` trong mảng `self.data`.
        idx = self.write + self.capacity - 1

        self.data[self.write] = data  # Lưu trữ dữ liệu vào mảng `self.data`.
        self.update(idx, priority)  # Cập nhật ưu tiên của node lá và lan truyền thay đổi.

        self.write = (self.write + 1) % self.capacity  # Cập nhật con trỏ ghi (buffer vòng).
        if self.n_entries < self.capacity:  # Nếu cây chưa đầy,
            self.n_entries += 1  # tăng số lượng mẫu.

    def update(self, idx: int, priority: float) -> None:
        """
        Cập nhật ưu tiên của một node lá đã có trong cây.

        Args:
            idx (int): Chỉ mục của node lá trong mảng `self.tree` cần được cập nhật.
            priority (float): Giá trị ưu tiên mới.
        """
        change = priority - self.tree[idx]  # Tính lượng thay đổi của ưu tiên.
        self.tree[idx] = priority  # Cập nhật giá trị ưu tiên của node lá.
        self._propagate(idx, change)  # Lan truyền thay đổi lên các node cha.

    def get(self, s: float) -> Tuple[int, float, object]:
        """
        Lấy mẫu dữ liệu từ cây dựa trên giá trị `s` (được lấy mẫu từ [0, total_priority]).

        Args:
            s (float): Giá trị được lấy mẫu.

        Returns:
            Tuple[int, float, object]:
                - Chỉ mục của node lá được lấy mẫu trong mảng `self.tree`.
                - Ưu tiên của mẫu được lấy mẫu.
                - Dữ liệu của mẫu được lấy mẫu.
        """
        idx = self._retrieve(0, s)  # Tìm chỉ mục node lá tương ứng với `s`.
        # `dataIdx` là chỉ mục của dữ liệu trong mảng `self.data` tương ứng với node lá `idx`.
        dataIdx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[dataIdx]


class PrioritizedReplayBuffer:
    """
    Replay buffer với việc lấy mẫu ưu tiên dựa trên TD error.
    Các transition có TD error cao hơn (nghĩa là "gây ngạc nhiên" hơn cho agent)
    sẽ có xác suất được lấy mẫu cao hơn.
    Điều này giúp agent học hiệu quả hơn từ các kinh nghiệm quan trọng.
    Sử dụng SumTree để quản lý ưu tiên.
    """
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001, epsilon: float = 0.01) -> None:
        """
        Khởi tạo PrioritizedReplayBuffer.

        Args:
            capacity (int): Kích thước tối đa của buffer.
            alpha (float): Tham số ưu tiên. alpha=0 tương ứng với lấy mẫu đều (uniform sampling).
                           alpha=1 tương ứng với lấy mẫu hoàn toàn dựa trên ưu tiên.
                           Giá trị phổ biến: 0.5 - 0.7.
            beta (float): Tham số cho importance sampling (IS) weight. Bắt đầu từ `beta` và
                          tăng dần lên 1.0 trong quá trình huấn luyện.
                          Dùng để bù đắp cho sự thiên lệch do lấy mẫu ưu tiên.
                          Giá trị phổ biến ban đầu: 0.4 - 0.6.
            beta_increment (float): Lượng tăng `beta` sau mỗi lần lấy mẫu.
            epsilon (float): Một giá trị nhỏ (ví dụ: 0.01) được thêm vào TD error để đảm bảo rằng
                             ngay cả các transition có TD error bằng 0 cũng có một xác suất nhỏ được lấy mẫu.
                             Nó cũng ngăn chặn ưu tiên bằng 0.
        """
        self.tree = SumTree(capacity)  # Khởi tạo SumTree với capacity đã cho.
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0  # Ưu tiên tối đa ban đầu, dùng để gán cho các transition mới.
                                  # Đảm bảo các transition mới có cơ hội được chọn cao.

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Thêm một transition mới vào buffer.
        Transition mới được gán ưu tiên tối đa hiện tại để đảm bảo chúng được xem xét sớm.

        Args:
            state (np.ndarray): State hiện tại.
            action (int): Hành động đã thực hiện.
            reward (float): Phần thưởng nhận được.
            next_state (np.ndarray): State tiếp theo.
            done (bool): True nếu episode kết thúc, False nếu không.
        """
        transition = (state, action, reward, next_state, done) # Đóng gói transition thành một tuple.

        # Tính toán ưu tiên thực tế để lưu vào SumTree.
        # p_i = (TD_error_i + epsilon)^alpha. Đối với transition mới, chúng ta không có TD error,
        # vì vậy gán cho nó ưu tiên tối đa hiện tại (`self.max_priority`).
        # `self.max_priority` thường là giá trị (abs(td_error) + epsilon)**alpha lớn nhất đã thấy.
        priority = self.max_priority ** self.alpha # Trong một số triển khai, ưu tiên cho transition mới là max_priority^alpha
                                          # Ở đây, nó nên là giá trị ưu tiên đã được nâng lên lũy thừa alpha.
                                          # Nếu self.max_priority đã là (abs(td_error)+eps)^alpha, thì dòng này đúng.
                                          # Nếu self.max_priority là abs(td_error)+eps, thì cần (self.max_priority)**self.alpha

        # Sửa lại: Khi thêm mới, chúng ta chưa biết TD-error, nên gán ưu tiên lớn nhất đã thấy.
        # self.max_priority lưu trữ giá trị (abs(td_error) + epsilon) ** self.alpha lớn nhất.
        # Nên khi thêm mới, chỉ cần dùng self.max_priority.
        self.tree.add(priority, transition) # Thêm vào SumTree với ưu tiên tối đa.

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], np.ndarray]:
        """
        Lấy một batch các transition từ buffer theo phân phối ưu tiên.
        Cũng tính toán importance sampling (IS) weights.

        Args:
            batch_size (int): Số lượng transition cần lấy mẫu.

        Returns:
            Tuple:
                - states (np.ndarray): Batch các state.
                - actions (np.ndarray): Batch các hành động.
                - rewards (np.ndarray): Batch các phần thưởng.
                - next_states (np.ndarray): Batch các state tiếp theo.
                - dones (np.ndarray): Batch các cờ done.
                - indices (List[int]): Danh sách các chỉ mục của các transition được lấy mẫu trong SumTree.
                                      Dùng để cập nhật ưu tiên sau này.
                - weights (np.ndarray): Các importance sampling weights cho mỗi transition trong batch.
        """
        batch = []  # Danh sách để lưu trữ các transition được lấy mẫu.
        indices = []  # Danh sách để lưu trữ các chỉ mục của SumTree.
        priorities_sampled = [] # Danh sách để lưu trữ ưu tiên của các mẫu được chọn

        # Chia tổng ưu tiên (self.tree.total()) thành `batch_size` đoạn (segment) bằng nhau.
        # Chúng ta sẽ lấy mẫu một giá trị `s` từ mỗi đoạn này.
        segment = self.tree.total() / batch_size

        # Tăng `beta` (annealing beta towards 1.0).
        # `beta` được sử dụng trong tính toán IS weights.
        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            # Xác định khoảng [a, b] cho đoạn thứ i.
            a = segment * i
            b = segment * (i + 1)
            # Lấy mẫu một giá trị `s` ngẫu nhiên từ khoảng [a, b].
            # Điều này đảm bảo rằng các mẫu được chọn từ các phần khác nhau của phân phối ưu tiên,
            # giúp giảm phương sai.
            s_sample = random.uniform(a, b)

            # Lấy (chỉ mục, ưu tiên, dữ liệu) từ SumTree dựa trên `s_sample`.
            idx, priority, data = self.tree.get(s_sample)
            priorities_sampled.append(priority)
            batch.append(data)
            indices.append(idx)

        # Tính toán xác suất lấy mẫu P(i) cho mỗi transition i trong batch.
        # P(i) = p_i / sum_k(p_k), trong đó p_i là ưu tiên của transition i.
        sampling_probabilities = np.array(priorities_sampled) / self.tree.total()

        # Tính toán importance sampling (IS) weights.
        # w_i = (N * P(i)) ^ (-beta)
        # N là số lượng entry trong buffer (`self.tree.n_entries`).
        # IS weights được sử dụng để điều chỉnh gradient updates nhằm bù đắp cho sự thiên lệch
        # gây ra bởi việc lấy mẫu ưu tiên.
        weights = (self.tree.n_entries * sampling_probabilities) ** (-self.beta)
        # Chuẩn hóa weights bằng cách chia cho weight lớn nhất.
        # Điều này giúp ổn định quá trình học, vì nó đảm bảo rằng các weight nằm trong khoảng [0, 1]
        # và các gradient update không bị phóng đại quá mức.
        weights = weights / weights.max()

        # Tách batch các tuple transition thành các mảng riêng biệt cho states, actions, etc.
        states, actions, rewards, next_states, dones = zip(*batch)

        # Chuyển đổi các danh sách thành mảng NumPy với kiểu dữ liệu phù hợp.
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.uint8), # Sử dụng uint8 cho cờ boolean (0 hoặc 1)
            indices,
            np.array(weights, dtype=np.float32)
        )

    def update_priorities(self, indices: List[int], td_errors: np.ndarray) -> None:
        """
        Cập nhật ưu tiên của các transition đã được lấy mẫu dựa trên TD error mới của chúng.

        Args:
            indices (List[int]): Danh sách các chỉ mục của SumTree (nhận được từ hàm `sample`).
            td_errors (np.ndarray): Mảng các TD error (giá trị tuyệt đối) tương ứng với các transition.
        """
        for idx, td_error in zip(indices, td_errors):
            # Tính toán ưu tiên mới: p_i = (|TD_error_i| + epsilon) ^ alpha.
            # `self.epsilon` đảm bảo rằng ngay cả khi TD error bằng 0, ưu tiên vẫn khác 0.
            # `self.alpha` kiểm soát mức độ ưu tiên.
            priority = (abs(td_error) + self.epsilon) ** self.alpha

            # Cập nhật `self.max_priority` nếu ưu tiên mới này lớn hơn.
            # Điều này đảm bảo rằng các transition mới được thêm vào sẽ có ưu tiên cao.
            # Lưu ý: self.max_priority nên lưu giá trị *sau khi* đã nâng lên lũy thừa alpha.
            self.max_priority = max(self.max_priority, priority)

            # Cập nhật ưu tiên của transition trong SumTree.
            self.tree.update(idx, priority)

    def __len__(self) -> int:
        """Trả về số lượng transition hiện có trong buffer."""
        return self.tree.n_entries