import gymnasium as gym  # OpenAI Gym, a toolkit for developing and comparing reinforcement learning algorithms.
import numpy as np  # NumPy for numerical operations, especially array manipulation.
import cv2  # OpenCV library for image processing tasks like resizing and color conversion.
from collections import deque  # `deque` (double-ended queue) for efficiently managing a fixed-size list of frames.

class FrameStack:
    """
    Class quản lý stack các frame liên tiếp.
    Trong nhiều trò chơi RL, một frame đơn lẻ không cung cấp đủ thông tin về động lực học
    (ví dụ: hướng di chuyển). Việc xếp chồng nhiều frame liên tiếp giúp agent nhận biết được chuyển động.
    """

    def __init__(self, num_frames=4):
        """
        Khởi tạo frame stack.

        Args:
            num_frames (int): số frame để stack (ví dụ: 4 frame).
        """
        self.num_frames = num_frames  # Số lượng frame tối đa được giữ trong stack.
        # Khởi tạo một deque với độ dài tối đa là `num_frames`.
        # Khi deque đầy và một phần tử mới được thêm vào, phần tử cũ nhất sẽ tự động bị loại bỏ.
        self.frames = deque(maxlen=num_frames)

    def reset(self, frame):
        """
        Reset frame stack với frame mới.
        Điều này thường được gọi khi bắt đầu một episode mới. Frame đầu tiên của episode
        sẽ được sao chép để lấp đầy toàn bộ stack.

        Args:
            frame (np.ndarray): Frame đầu tiên (đã được xử lý) của một episode.

        Returns:
            np.ndarray: State được tạo thành từ các frame đã stack (frame đầu tiên được lặp lại).
        """
        # Xóa tất cả các frame hiện có trong deque.
        self.frames.clear()
        # Lấp đầy deque bằng cách thêm `frame` ban đầu `num_frames` lần.
        for _ in range(self.num_frames):
            self.frames.append(frame)
        # Trả về state hiện tại (là một mảng NumPy của các frame đã stack).
        return self._get_state()

    def add(self, frame):
        """
        Thêm frame mới vào stack.
        Frame mới nhất được thêm vào, và nếu stack đã đầy, frame cũ nhất sẽ bị loại bỏ.

        Args:
            frame (np.ndarray): Frame mới (đã được xử lý) để thêm vào stack.

        Returns:
            np.ndarray: State được cập nhật từ các frame đã stack.
        """
        # Thêm frame mới vào cuối deque.
        self.frames.append(frame)
        # Trả về state hiện tại.
        return self._get_state()

    def _get_state(self):
        """
        Lấy state từ các frame đã stack.
        Chuyển đổi deque các frame thành một mảng NumPy duy nhất.
        Thứ tự các frame trong mảng thường là (num_frames, height, width).

        Returns:
            np.ndarray: Mảng NumPy đại diện cho state, được tạo từ các frame trong deque.
        """
        # Chuyển đổi deque `self.frames` thành một mảng NumPy.
        # Các frame được xếp chồng dọc theo chiều thứ nhất (axis 0).
        return np.array(self.frames)

class FrameProcessor:
    """
    Class xử lý từng frame riêng lẻ từ môi trường.
    Xử lý bao gồm chuyển đổi sang thang độ xám và thay đổi kích thước để giảm độ phức tạp tính toán
    và chuẩn hóa đầu vào cho mạng neural.
    """

    def __init__(self, frame_size=(84, 84)):
        """
        Khởi tạo FrameProcessor.

        Args:
            frame_size (tuple): Kích thước mong muốn của frame sau khi xử lý (height, width).
                                Kích thước (84, 84) là phổ biến trong các tài liệu DQN.
        """
        self.frame_size = frame_size  # Lưu trữ kích thước frame mục tiêu.

    def process(self, frame):
        """
        Xử lý một frame.

        Args:
            frame (np.ndarray): Frame gốc từ môi trường (thường là RGB).

        Returns:
            np.ndarray: Frame đã được xử lý (thang độ xám, đã thay đổi kích thước, kiểu uint8).
        """

        # Chuyển đổi sang thang độ xám nếu frame là ảnh màu (3 kênh).
        # Kiểm tra xem frame có 3 chiều và chiều cuối cùng (kênh màu) có kích thước là 3 (RGB).
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # Chuyển từ RGB sang thang độ xám.
        # Nếu frame đã là thang độ xám (ví dụ, shape là (height, width) hoặc (height, width, 1)),
        # bước này có thể không cần thiết hoặc cần điều chỉnh.

        # Resize frame về kích thước `self.frame_size`.
        # `cv2.INTER_AREA` thường được khuyến nghị khi thu nhỏ ảnh để tránh răng cưa.
        frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_AREA)

        # Chuyển đổi kiểu dữ liệu của frame sang `np.uint8` (số nguyên không dấu 8-bit).
        # Giá trị pixel sẽ nằm trong khoảng [0, 255]. Điều này quan trọng cho việc lưu trữ bộ nhớ
        # và là định dạng đầu vào phổ biến cho các mạng CNN.
        return frame.astype(np.uint8)

class PacmanEnv:
    """
    Môi trường wrapper cho Pacman, kết hợp xử lý frame và frame stacking.
    Lớp này tạo ra một giao diện chuẩn hóa giống Gym cho môi trường Pacman đã được tiền xử lý.
    """

    def __init__(self, env_name="ALE/Pacman-v5", render_mode=None, stack_frames=4): # Sửa tên môi trường thành chuẩn Atari
        """
        Khởi tạo môi trường Pacman.

        Args:
            env_name (str): Tên của môi trường Gym (ví dụ: "ALE/Pacman-v5" cho Atari Pacman).
            render_mode (str, optional): Chế độ render của Gym (ví dụ: "human", "rgb_array").
            stack_frames (int): Số lượng frame để stack làm state.
        """
        # Khởi tạo môi trường gốc của Gym.
        self.env = gym.make(env_name, render_mode=render_mode)

        # Khởi tạo FrameProcessor để xử lý từng frame.
        self.processor = FrameProcessor(frame_size=(84, 84)) # Kích thước frame chuẩn.
        # Khởi tạo FrameStack để quản lý việc xếp chồng các frame đã xử lý.
        self.stacker = FrameStack(num_frames=stack_frames)

        # Lưu trữ không gian hành động từ môi trường gốc.
        # Agent sẽ chọn hành động từ không gian này.
        self.action_space = self.env.action_space

        # Định nghĩa không gian quan sát (observation space) của môi trường wrapper.
        # Đây là hình dạng của state mà agent sẽ nhận được sau khi xử lý và xếp chồng frame.
        # `low=0`, `high=255` vì các frame là ảnh thang độ xám 8-bit.
        # `shape=(stack_frames, 84, 84)`: (số frame stack, chiều cao, chiều rộng).
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(stack_frames, 84, 84), # (channels, height, width) - PyTorch convention
            dtype=np.uint8
        )

    def reset(self):
        """
        Reset môi trường và trả về state ban đầu đã được xử lý và stack.

        Returns:
            tuple:
                - np.ndarray: State ban đầu (các frame đã stack).
                - dict: Thông tin bổ sung từ môi trường (info).
        """
        # Reset môi trường Gym gốc, nhận observation ban đầu và thông tin.
        obs, info = self.env.reset()
        # Xử lý frame observation ban đầu (chuyển sang thang độ xám, resize).
        processed_obs = self.processor.process(obs)
        # Reset frame stacker với frame đầu tiên đã xử lý.
        # Frame này sẽ được lặp lại `stack_frames` lần để tạo state ban_đầu.
        stacked_state = self.stacker.reset(processed_obs)
        return stacked_state, info

    def step(self, action):
        """
        Thực hiện một hành động trong môi trường.

        Args:
            action (int): Hành động được chọn bởi agent.

        Returns:
            tuple:
                - np.ndarray: State tiếp theo (các frame đã stack).
                - float: Phần thưởng (reward) nhận được.
                - bool: `terminated` (True nếu episode kết thúc do điều kiện của game, ví dụ: hết mạng).
                - bool: `truncated` (True nếu episode kết thúc do giới hạn thời gian hoặc điều kiện bên ngoài).
                - dict: Thông tin bổ sung (info).
        """
        # Thực hiện hành động trong môi trường Gym gốc.
        obs, reward, terminated, truncated, info = self.env.step(action)

        # (Tùy chọn) Reward shaping: Điều chỉnh reward gốc.
        # Ví dụ: thêm một phần thưởng nhỏ cho mỗi bước để khuyến khích agent sống sót lâu hơn.
        # Tuy nhiên, reward shaping cần cẩn thận để không làm thay đổi mục tiêu của bài toán.
        shaped_reward = reward # + 0.01 # Bỏ comment nếu muốn thử nghiệm reward shaping

        # Xử lý frame observation mới nhận được.
        processed_obs = self.processor.process(obs)

        # Thêm frame đã xử lý vào stacker để tạo state tiếp theo.
        stacked_state = self.stacker.add(processed_obs)
        return stacked_state, shaped_reward, terminated, truncated, info

    def render(self):
        """Render môi trường (nếu render_mode được thiết lập)."""
        # Gọi phương thức render của môi trường Gym gốc.
        return self.env.render()

    def close(self):
        """Đóng môi trường và giải phóng tài nguyên."""
        # Gọi phương thức close của môi trường Gym gốc.
        self.env.close()