# env.py
# Dòng này là một comment, chỉ ra tên của file là env.py. Nó không có tác dụng thực thi.

import gymnasium as gym
# Dòng này import thư viện `gymnasium`. Gymnasium là một bộ công cụ (toolkit) tiêu chuẩn 
# để phát triển và so sánh các thuật toán học tăng cường (Reinforcement Learning - RL). 
# Nó cung cấp một tập hợp các môi trường mô phỏng (như game Atari, các bài toán robot...).


import numpy as np
# Dòng này import thư viện `numpy`, một thư viện nền tảng cho tính toán khoa học 
# trong Python, đặc biệt mạnh về xử lý mảng đa chiều (arrays). 
# Trong ngữ cảnh này, numpy sẽ được dùng để xử lý các frame (hình ảnh) của game, 
# vì chúng thường được biểu diễn dưới dạng mảng số (pixel).
# `as np` cũng là một bí danh phổ biến cho numpy.

import cv2
# Dòng này import thư viện `cv2`, là thư viện OpenCV (Open Source Computer Vision Library). 
# OpenCV cung cấp rất nhiều hàm cho xử lý ảnh và video. Ở đây, nó sẽ được dùng để:
# 1. Chuyển đổi ảnh màu sang ảnh xám (grayscale).
# 2. Thay đổi kích thước (resize) ảnh.
# Việc này giúp giảm độ phức tạp của dữ liệu đầu vào cho mô hình RL.

from collections import deque
# Dòng này import lớp `deque` từ module `collections`. 
# Nó là một cấu trúc dữ liệu giống list nhưng hiệu quả hơn cho việc thêm/xóa 
# phần tử ở cả đầu và cuối hàng đợi. 
# Đặc biệt, `deque` có thể được giới hạn kích thước tối đa (`maxlen`). Khi một phần tử mới 
# được thêm vào deque đã đầy, phần tử ở đầu đối diện sẽ tự động bị loại bỏ. 
# Điều này rất lý tưởng để lưu trữ một số lượng frame gần nhất (frame stacking).

class FrameStack:
# Dòng này khai báo một lớp (class) tên là `FrameStack`. 
# Mục đích của lớp này là để quản lý việc lưu trữ và kết hợp một chuỗi các frame 
# (khung hình) liên tiếp từ môi trường game. Việc "stack" (xếp chồng) các frame 
# giúp cho mô hình RL có thể nhận biết được thông tin về chuyển động (ví dụ: hướng di chuyển
# của bóng, ma...).

    """Class quản lý stack các frame liên tiếp"""
    # Đây là một docstring (chuỗi tài liệu), mô tả ngắn gọn mục đích của lớp `FrameStack`.

    def __init__(self, num_frames=4):
    # Đây là hàm khởi tạo (constructor) của lớp `FrameStack`, được gọi khi một đối tượng
    # `FrameStack` mới được tạo ra.
    # `self` là tham chiếu đến chính đối tượng đang được tạo.
    # `num_frames=4`: là một tham số đầu vào, chỉ định số lượng frame cần stack. 
    # Giá trị mặc định là 4 (một con số phổ biến trong các nghiên cứu DQN cho Atari).

        

        self.num_frames = num_frames

        self.frames = deque(maxlen=num_frames)
        # Dòng này tạo ra một đối tượng `deque` và gán nó cho biến instance `self.frames`.
        # `maxlen=num_frames` là phần quan trọng: nó giới hạn kích thước của deque 
        # bằng đúng số frame cần stack. Khi deque đầy và một frame mới được thêm vào, 
        # frame cũ nhất sẽ tự động bị loại bỏ.

    def reset(self, frame):
    # Đây là một phương thức (method) của lớp `FrameStack`, thường được gọi khi 
    # môi trường game được reset (bắt đầu một episode mới).
    # `self`: tham chiếu đến đối tượng `FrameStack`.
    # `frame`: frame đầu tiên của episode mới (đã qua xử lý).

        """Reset frame stack với frame mới"""
        # Docstring cho phương thức `reset`.

        for _ in range(self.num_frames):
        # Dòng này bắt đầu một vòng lặp chạy `self.num_frames` lần (ví dụ: 4 lần).

            self.frames.append(frame)
            # Trong mỗi lần lặp, dòng này thêm chính cái `frame` đầu tiên đó vào `self.frames` (deque).
            # Mục đích: Khi bắt đầu một game mới, không có các frame trước đó. 
            # Để đảm bảo state ban đầu có đúng kích thước (ví dụ: 4x84x84), 
            # chúng ta điền đầy deque bằng cách lặp lại frame đầu tiên.

        return self._get_state()
        # Dòng này gọi phương thức nội bộ `_get_state()` (sẽ giải thích sau) 
        # để lấy trạng thái (state) được tạo từ các frame vừa được thêm vào deque 
        # và trả về trạng thái đó.

    def add(self, frame):
    # Đây là phương thức được gọi sau mỗi bước (step) trong môi trường game 
    # để thêm frame mới nhất vào stack.
    # `self`: tham chiếu đến đối tượng `FrameStack`.
    # `frame`: frame mới nhất nhận được từ môi trường (đã qua xử lý).

        """Thêm frame mới vào stack"""
        # Docstring cho phương thức `add`.

        self.frames.append(frame)
        # Dòng này thêm `frame` mới vào cuối (bên phải) của deque `self.frames`.
        # Do deque có `maxlen`, nếu nó đã đầy, frame cũ nhất ở đầu (bên trái) sẽ tự động bị loại bỏ.

        return self._get_state()
        # Dòng này gọi `_get_state()` để lấy trạng thái được cập nhật từ deque 
        # (bao gồm frame mới và loại bỏ frame cũ nhất nếu cần) và trả về trạng thái đó.

    def _get_state(self):
    # Đây là một phương thức "nội bộ" (internal method - thường được đánh dấu bằng dấu gạch dưới `_` ở đầu).
    # Nó không dùng để gọi trực tiếp từ bên ngoài lớp, mà được các phương thức khác (`reset`, `add`) sử dụng.
    # Mục đích: chuyển đổi deque chứa các frame riêng lẻ thành một cấu trúc dữ liệu duy nhất 
    # (thường là một mảng NumPy) đại diện cho trạng thái hiện tại.
    # `self`: tham chiếu đến đối tượng `FrameStack`.

        """Lấy state từ các frame đã stack"""
        # Docstring cho phương thức `_get_state`.

        return np.array(self.frames)
        # Dòng này chuyển đổi đối tượng `deque` (`self.frames`), vốn chứa các mảng 2D (các frame đã xử lý), 
        # thành một mảng NumPy duy nhất. 
        # Nếu `num_frames=4` và mỗi frame là 84x84, kết quả trả về sẽ là một mảng NumPy 
        # có hình dạng (shape) là `(4, 84, 84)`. Đây chính là "state" (trạng thái) 
        # mà mô hình RL sẽ sử dụng làm đầu vào.

class FrameProcessor:
# Dòng này khai báo một lớp tên là `FrameProcessor`.
# Mục đích của lớp này là thực hiện các bước tiền xử lý (preprocessing) cần thiết trên 
# mỗi frame nhận được từ môi trường game gốc trước khi đưa vào `FrameStack`.
# Tiền xử lý thường bao gồm chuyển sang ảnh xám và thay đổi kích thước.

    """Class xử lý frame"""
    # Docstring mô tả lớp `FrameProcessor`.

    def __init__(self, frame_size=(84, 84)):
    # Hàm khởi tạo của lớp `FrameProcessor`.
    # `self`: tham chiếu đến đối tượng.
    # `frame_size=(84, 84)`: tham số đầu vào chỉ định kích thước mong muốn của frame sau khi xử lý. 
    # Kích thước (84, 84) là tiêu chuẩn từ bài báo DQN gốc của DeepMind.

        """
        Khởi tạo frame processor
        
        Args:
            frame_size: kích thước frame sau khi resize
        """
        # Docstring cho hàm `__init__`.

        self.frame_size = frame_size
        # Lưu kích thước frame mong muốn vào biến instance `self.frame_size`.

    def process(self, frame):
    # Phương thức chính của lớp này, thực hiện việc xử lý một frame đầu vào.
    # `self`: tham chiếu đến đối tượng.
    # `frame`: frame gốc nhận từ môi trường (thường là một mảng NumPy RGB).

        """
        Xử lý một frame
        
        Args:
            frame: array RGB (hoặc Grayscale nếu môi trường trả về vậy)
            
        Returns:
            numpy array: frame đã được xử lý (grayscale, resized)
        """
        # Docstring cho phương thức `process`. Chú ý: tên tham số trong comment gốc là `framebatch-size` có vẻ là lỗi đánh máy, nên là `frame`.

        # Chuyển thành grayscale
        if len(frame.shape) == 3 and frame.shape[2] == 3:
        # Dòng này kiểm tra xem frame đầu vào có phải là ảnh màu RGB hay không.
        # `len(frame.shape) == 3`: Kiểm tra xem mảng có 3 chiều (height, width, channels) không.
        # `frame.shape[2] == 3`: Kiểm tra xem chiều thứ 3 (channels) có kích thước là 3 (cho R, G, B) không.
        # Điều này để tránh lỗi nếu frame đầu vào đã là ảnh xám (chỉ có 2 chiều).

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            # Nếu frame là ảnh màu RGB, dòng này sử dụng hàm `cvtColor` của OpenCV 
            # để chuyển đổi nó sang ảnh xám (grayscale). 
            # `cv2.COLOR_RGB2GRAY` là mã chỉ định loại chuyển đổi.
            # Việc chuyển sang ảnh xám giúp giảm lượng dữ liệu cần xử lý mà thường không làm mất 
            # thông tin quan trọng cho việc chơi game.

        # Resize
        frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_AREA)
        # Dòng này sử dụng hàm `resize` của OpenCV để thay đổi kích thước của frame 
        # (dù là gốc hay đã chuyển sang xám) về kích thước mong muốn `self.frame_size` (ví dụ: (84, 84)).
        # `interpolation=cv2.INTER_AREA`: chỉ định phương pháp nội suy (interpolation) được sử dụng 
        # khi thay đổi kích thước. `INTER_AREA` thường được khuyên dùng khi thu nhỏ ảnh 
        # vì nó giúp giảm hiện tượng răng cưa (aliasing).

        return frame
        # Trả về frame đã được xử lý (đã chuyển sang xám nếu cần và đã resize). 
        # Đây sẽ là frame được đưa vào `FrameStack`.

class PacmanEnv:
# Dòng này khai báo lớp `PacmanEnv`. Đây là lớp chính, đóng vai trò như một "wrapper" 
# (lớp bao bọc) xung quanh môi trường Pacman gốc của `gymnasium`.
# Mục đích: Tạo ra một giao diện môi trường tuân thủ chuẩn của `gymnasium` 
# nhưng cung cấp các quan sát (observations) đã được tiền xử lý và xếp chồng frame.

    """Môi trường wrapper cho Pacman"""
    # Docstring mô tả lớp `PacmanEnv`.

    def __init__(self, env_name="Pacman-v0", render_mode=None, stack_frames=4):
    # Hàm khởi tạo của lớp `PacmanEnv`.
    # `self`: tham chiếu đến đối tượng.
    # `env_name="Pacman-v0"`: Tên của môi trường gốc trong `gymnasium` muốn sử dụng. 
    # Mặc định là "Pacman-v0". Có thể thay đổi để dùng với game Atari khác nếu cần.
    # `render_mode=None`: Chế độ hiển thị hình ảnh của môi trường (ví dụ: 'human' để hiện cửa sổ game, 
    # 'rgb_array' để trả về ảnh dưới dạng mảng NumPy). `None` nghĩa là không render. 
    # Giá trị này sẽ được truyền xuống môi trường gốc.
    # `stack_frames=4`: Số lượng frame cần stack, giá trị này sẽ được dùng để khởi tạo `FrameStack`.

        """
        Khởi tạo môi trường Pacman
        
        Args:
            env_name: tên môi trường
            render_mode: chế độ render
            stack_frames: số frame để stack
        """
        # Docstring cho hàm `__init__`.

        # Khởi tạo môi trường gốc
        self.env = gym.make(env_name, render_mode=render_mode)
        # Dòng này tạo ra một instance của môi trường `gymnasium` gốc (ví dụ: Pacman) 
        # bằng cách sử dụng hàm `gym.make()`. Tên môi trường và chế độ render được truyền vào.
        # Instance này được lưu vào biến `self.env`. Lớp `PacmanEnv` sẽ tương tác với 
        # môi trường gốc này bên trong.

        # Frame processor và stacker
        self.processor = FrameProcessor(frame_size=(84, 84))
        # Tạo một instance của lớp `FrameProcessor` đã định nghĩa ở trên. 
        # Kích thước frame đích (84, 84) được hardcode ở đây (phổ biến cho DQN).
        # Instance này được lưu vào `self.processor`.

        self.stacker = FrameStack(num_frames=stack_frames)
        # Tạo một instance của lớp `FrameStack` đã định nghĩa ở trên. 
        # Số lượng frame cần stack (`num_frames`) được lấy từ tham số `stack_frames` của hàm `__init__`.
        # Instance này được lưu vào `self.stacker`.

        # Lưu trữ không gian hành động và quan sát
        self.action_space = self.env.action_space
        # Dòng này lấy thông tin về không gian hành động (các hành động hợp lệ mà agent có thể thực hiện) 
        # từ môi trường gốc (`self.env.action_space`) và gán nó cho thuộc tính `action_space` 
        # của lớp wrapper (`self.action_space`). Lớp wrapper không thay đổi các hành động có thể thực hiện.

        # Định nghĩa không gian quan sát mới sau khi stack frame
        self.observation_space = gym.spaces.Box(
        # Dòng này định nghĩa lại không gian quan sát (observation space) cho lớp wrapper. 
        # Vì wrapper trả về các frame đã được xử lý và xếp chồng, không gian quan sát 
        # của nó khác với môi trường gốc.
        # `gym.spaces.Box` được sử dụng để định nghĩa một không gian gồm các giá trị số thực 
        # (hoặc nguyên) trong một khoảng giới hạn, có hình dạng (shape) nhất định.

            low=0, 
            # Giá trị pixel thấp nhất có thể có (thường là 0 cho ảnh xám).
            high=255,
            # Giá trị pixel cao nhất có thể có (thường là 255 cho ảnh 8-bit).
            shape=(stack_frames, 84, 84),
            # Hình dạng (shape) của một quan sát (observation) mà môi trường wrapper này trả về.
            # Nó là một mảng 3 chiều: (số_frame_đã_stack, chiều_cao_frame, chiều_rộng_frame).
            # Giá trị `stack_frames` lấy từ tham số, (84, 84) là kích thước sau khi xử lý.
            dtype=np.uint8
            # Kiểu dữ liệu của các giá trị pixel trong mảng quan sát. `np.uint8` là số nguyên 
            # không dấu 8-bit (giá trị từ 0 đến 255), phù hợp cho ảnh xám.
        )
    
    def reset(self):
    # Phương thức `reset` của lớp wrapper. Nó phải tuân theo giao diện của `gymnasium`.
    # Được gọi khi bắt đầu một episode mới.
    # `self`: tham chiếu đến đối tượng.

        """Reset môi trường"""
        # Docstring cho phương thức `reset`.

        obs, info = self.env.reset()
        # Gọi phương thức `reset()` của môi trường gốc (`self.env`). 
        # Nó trả về quan sát (frame) thô đầu tiên (`obs`) và một dictionary thông tin bổ sung (`info`).

        # Xử lý frame
        processed_obs = self.processor.process(obs)
        # Sử dụng `self.processor` để xử lý (chuyển xám, resize) cái frame thô `obs` đầu tiên.

        # Reset frame stacker với frame đầu tiên
        stacked_state = self.stacker.reset(processed_obs)
        # Sử dụng `self.stacker` để reset hàng đợi frame và điền đầy nó bằng frame đã xử lý đầu tiên.
        # Kết quả trả về là trạng thái ban đầu đã được stack (`stacked_state`).

        return stacked_state, info
        # Trả về trạng thái đã stack (`stacked_state`) và dictionary `info`. 
        # Đây là đầu ra chuẩn của phương thức `reset` trong `gymnasium`, nhưng với 
        # quan sát đã được biến đổi bởi wrapper.

    def step(self, action):
    # Phương thức `step` của lớp wrapper. Nó cũng phải tuân theo giao diện `gymnasium`.
    # Được gọi sau khi agent chọn một hành động để thực hiện trong môi trường.
    # `self`: tham chiếu đến đối tượng.
    # `action`: hành động mà agent đã chọn.

        """Thực hiện một hành động trong môi trường"""
        # Docstring cho phương thức `step`.

        obs, reward, terminated, truncated, info = self.env.step(action)
        # Thực thi hành động `action` trong môi trường gốc (`self.env`) bằng cách gọi `step()` của nó.
        # Môi trường gốc trả về:
        # - `obs`: quan sát (frame) thô tiếp theo.
        # - `reward`: phần thưởng nhận được sau hành động.
        # - `terminated`: boolean, True nếu episode kết thúc do đạt mục tiêu hoặc thất bại (ví dụ: hết mạng).
        # - `truncated`: boolean, True nếu episode kết thúc do giới hạn thời gian hoặc lý do khác không tự nhiên.
        # - `info`: dictionary thông tin bổ sung.

        # Xử lý frame
        processed_obs = self.processor.process(obs)
        # Xử lý frame thô `obs` mới nhận được bằng `self.processor`.

        # Thêm frame mới vào stack
        stacked_state = self.stacker.add(processed_obs)
        # Thêm frame đã xử lý mới vào `self.stacker`. `stacker.add()` sẽ tự động loại bỏ 
        # frame cũ nhất nếu cần và trả về trạng thái mới đã được stack (`stacked_state`).

        return stacked_state, reward, terminated, truncated, info
        # Trả về các giá trị theo chuẩn `gymnasium`: trạng thái đã stack mới (`stacked_state`), 
        # phần thưởng (`reward`), các cờ kết thúc (`terminated`, `truncated`), và `info`.
        # Quan trọng là nó trả về `stacked_state` thay vì `obs` thô.

    def render(self):
    # Phương thức `render` của wrapper. Dùng để hiển thị môi trường (nếu `render_mode` được đặt).
    # `self`: tham chiếu đến đối tượng.

        """Render môi trường"""
        # Docstring cho phương thức `render`.

        return self.env.render()
        # Đơn giản là gọi trực tiếp phương thức `render()` của môi trường gốc (`self.env`). 
        # Việc hiển thị thường dựa trên trạng thái gốc của game, không phải frame đã xử lý/stack.

    def close(self):
    # Phương thức `close` của wrapper. Dùng để giải phóng tài nguyên khi không sử dụng môi trường nữa.
    # `self`: tham chiếu đến đối tượng.

        """Đóng môi trường"""
        # Docstring cho phương thức `close`.

        self.env.close()
        # Gọi trực tiếp phương thức `close()` của môi trường gốc (`self.env`) để đảm bảo 
        # nó đóng các cửa sổ hiển thị hoặc giải phóng bộ nhớ đúng cách.