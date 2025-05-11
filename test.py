import argparse  # Thư viện để phân tích các đối số dòng lệnh.
import os        # Thư viện cung cấp các hàm để tương tác với hệ điều hành, ví dụ: tạo thư mục, nối đường dẫn.
import time      # Thư viện cung cấp các hàm liên quan đến thời gian, ví dụ: tạm dừng thực thi.
from typing import List # Type hinting cho danh sách.
from configs import Config # Import lớp Config chung từ file configs.py

import cv2  # OpenCV cho việc hiển thị frame nếu vừa render vừa record.
import matplotlib         # Thư viện chính để vẽ biểu đồ.
import matplotlib.pyplot as plt # Module con của Matplotlib cung cấp giao diện giống MATLAB để vẽ.
import numpy as np        # NumPy cho các phép toán số học, đặc biệt là tính toán thống kê.
import torch              # PyTorch cho deep learning.

# Đặt backend cho Matplotlib thành 'Agg'.
# 'Agg' là một backend không có giao diện người dùng đồ họa (GUI),
# nó chỉ ghi ra file. Điều này hữu ích khi chạy script trên server không có X server
# hoặc khi bạn chỉ muốn lưu biểu đồ vào file mà không hiển thị.
matplotlib.use('Agg')

from env import PacmanEnv  # Import lớp PacmanEnv đã được định nghĩa (wrapper cho môi trường Pacman).
from agent import DoubleDQNAgent  # Import lớp DoubleDQNAgent đã được định nghĩa.
from utils import set_seed, create_dirs, display_frames_as_gif # Import các hàm tiện ích.
from configs import TestConfig, Config # Import các lớp cấu hình TestConfig và Config chung.


def parse_args() -> argparse.Namespace:
    """Parse các tham số dòng lệnh cho script test."""
    parser = argparse.ArgumentParser(description="Test trained Double DQN for Pacman")

    # --- Các tham số liên quan đến Môi trường ---
    parser.add_argument("--env", type=str, default=Config.ENV_NAME, # Sử dụng giá trị từ Config làm mặc định
                        help="Pacman environment name (e.g., ALE/Pacman-v5)")
    parser.add_argument("--stack-frames", type=int, default=Config.STACK_FRAMES, # Sử dụng giá trị từ Config
                        help="Number of frames to stack as state")

    # --- Các tham số liên quan đến Model ---
    # `required=True if TestConfig.MODEL_PATH is None else False` nghĩa là:
    # Nếu TestConfig.MODEL_PATH không được định nghĩa sẵn trong file configs.py (là None),
    # thì tham số --model-path này là bắt buộc trên dòng lệnh.
    # Ngược lại, nếu đã có giá trị mặc định trong TestConfig, thì nó không bắt buộc.
    parser.add_argument("--model-path", type=str,
                        required=True if TestConfig.MODEL_PATH is None else False,
                        default=TestConfig.MODEL_PATH, # Sử dụng giá trị từ TestConfig
                        help="Path to the saved trained model checkpoint (.pth file)")

    # --- Các tham số liên quan đến Quá trình Test ---
    parser.add_argument("--episodes", type=int, default=TestConfig.TEST_EPISODES, # Sử dụng giá trị từ TestConfig
                        help="Number of episodes to run for testing")
    # `action="store_true"`: Nếu cờ này xuất hiện trên dòng lệnh (ví dụ: --render),
    # giá trị của args.render sẽ là True. Ngược lại, nó sẽ là giá trị default (ở đây là TestConfig.RENDER).
    parser.add_argument("--render", action="store_true", default=TestConfig.RENDER, # Sử dụng giá trị từ TestConfig
                        help="Render the environment during testing (display gameplay)")
    parser.add_argument("--record", action="store_true", default=TestConfig.RECORD, # Sử dụng giá trị từ TestConfig
                        help="Record video (GIF) of gameplay")
    parser.add_argument("--save-dir", type=str, default=TestConfig.VIDEO_DIR, # Sử dụng giá trị từ TestConfig
                        help="Directory to save recorded videos/GIFs")

    # --- Các tham số khác ---
    # Tự động chọn 'cuda' nếu có GPU, ngược lại là 'cpu'.
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for PyTorch computations (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=Config.SEED, # Sử dụng giá trị từ Config
                        help="Random seed for reproducibility")

    return parser.parse_args() # Phân tích và trả về các đối số.


def test(env: PacmanEnv, agent: DoubleDQNAgent, episodes: int, render: bool = False, record: bool = False,
         save_dir: str = 'videos') -> List[float]:
    """
    Test agent đã huấn luyện trên môi trường trong một số episode.

    Args:
        env (PacmanEnv): Môi trường Pacman đã được wrapper.
        agent (DoubleDQNAgent): Agent đã được load model.
        episodes (int): Số lượng episode để test.
        render (bool): Có hiển thị gameplay hay không.
        record (bool): Có ghi lại gameplay thành file GIF hay không.
        save_dir (str): Thư mục để lưu các file GIF đã ghi.

    Returns:
        List[float]: Danh sách các tổng phần thưởng đạt được trong mỗi episode.
    """
    rewards = []  # Danh sách để lưu tổng phần thưởng của mỗi episode.
    all_frames = [] # Danh sách để lưu tất cả các frame từ tất cả các episode (nếu record).

    # Biến này xác định xem có cần hiển thị frame bằng OpenCV hay không.
    # Chỉ True khi vừa muốn render (xem trực tiếp) VÀ vừa muốn record (lấy frame để lưu).
    show_frames = render and record

    for i in range(episodes):
        state, _ = env.reset()  # Reset môi trường để bắt đầu episode mới.
        done = False            # Cờ báo hiệu episode kết thúc do điều kiện của game (ví dụ: hết mạng).
        truncated = False       # Cờ báo hiệu episode kết thúc do giới hạn thời gian hoặc điều kiện bên ngoài.
        total_reward = 0        # Tổng phần thưởng của episode hiện tại.
        steps = 0               # Số bước trong episode hiện tại.
        episode_frames = []     # Danh sách để lưu các frame của episode hiện tại (nếu record).

        print(f"Episode {i + 1}/{episodes}")

        # Vòng lặp chính của một episode, tiếp tục cho đến khi episode kết thúc.
        while not (done or truncated):
            # Xử lý việc lấy frame để record hoặc render.
            if record:
                # Lấy frame hiện tại từ môi trường dưới dạng mảng numpy (RGB).
                # `env.render()` phải được gọi với `render_mode="rgb_array"` để trả về frame.
                frame = env.render()
                if frame is not None:
                    episode_frames.append(frame) # Thêm frame vào danh sách của episode.

                    # Nếu `show_frames` là True (nghĩa là vừa render vừa record),
                    # hiển thị frame bằng OpenCV.
                    if show_frames:
                        # Chuyển đổi frame từ RGB (chuẩn của Gym) sang BGR (chuẩn của OpenCV).
                        cv2.imshow("Pacman", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        # Đợi một khoảng thời gian ngắn (20ms) để frame được hiển thị và cho phép xử lý sự kiện cửa sổ.
                        cv2.waitKey(20)
            elif render:
                # Nếu chỉ render (không record), gọi `env.render()` mà không cần lấy frame.
                # `render_mode` của env có thể là "human".
                env.render()
                time.sleep(0.02)  # Tạm dừng một chút để người xem dễ theo dõi.

            # Agent chọn hành động dựa trên state hiện tại.
            # `training=False` để agent sử dụng chiến lược exploitation (chọn hành động tốt nhất),
            # không sử dụng epsilon-greedy exploration.
            action = agent.select_action(state, training=False)

            # Thực hiện hành động đã chọn trong môi trường.
            next_state, reward, done, truncated, _ = env.step(action)

            # Cập nhật các số liệu.
            total_reward += reward
            steps += 1

            # Chuyển sang state tiếp theo.
            state = next_state

        print(f"Episode {i + 1} finished with reward {total_reward:.1f} after {steps} steps")
        rewards.append(total_reward) # Lưu tổng phần thưởng của episode này.

        # Nếu record được bật và có frame đã thu thập, lưu episode hiện tại thành file GIF.
        if record and episode_frames:
            # Tạo tên file cho GIF, bao gồm số episode và phần thưởng.
            filename = os.path.join(save_dir, f"episode_{i + 1}_reward_{int(total_reward)}.gif")
            # Sử dụng hàm `display_frames_as_gif` (từ utils.py) để tạo và lưu GIF.
            display_frames_as_gif(episode_frames, filename=filename)
            print(f"Saved episode recording to {filename}")

            # Thêm các frame của episode này vào danh sách `all_frames`.
            all_frames.extend(episode_frames)

    # Đóng tất cả cửa sổ OpenCV nếu chúng đã được mở (`show_frames` là True).
    if show_frames:
        cv2.destroyAllWindows()

    # Nếu record được bật và có frame đã thu thập từ nhiều episode,
    # tạo một file GIF tổng hợp tất cả các episode.
    if record and all_frames:
        filename = os.path.join(save_dir, f"all_episodes.gif")
        # `all_frames[::5]` lấy 1 frame trong mỗi 5 frame để giảm kích thước file GIF.
        # `fps=20` đặt tốc độ khung hình của GIF là 20 frames per second.
        display_frames_as_gif(all_frames[::5], filename=filename, fps=20)
        print(f"Saved all episodes recording to {filename}")

    return rewards # Trả về danh sách các tổng phần thưởng.


def main() -> None:
    """Hàm chính để chạy quá trình test."""
    # Phân tích các đối số dòng lệnh.
    args = parse_args()

    # Đặt seed ngẫu nhiên để đảm bảo kết quả có thể tái tạo.
    set_seed(args.seed)

    # Tạo thư mục để lưu video/GIF nếu record được bật và thư mục chưa tồn tại.
    if args.record:
        create_dirs([args.save_dir])

    # Lấy tên môi trường từ đối số.
    env_name = args.env

    # Xác định `render_mode` cho môi trường dựa trên các cờ `render` và `record`.
    render_mode = None
    if args.render and args.record:
        # Nếu muốn xem trực tiếp (render) VÀ ghi lại (record),
        # `render_mode` phải là "rgb_array" để `env.render()` trả về frame.
        # Việc hiển thị sẽ được xử lý riêng bằng OpenCV trong hàm `test`.
        render_mode = "rgb_array"
    elif args.render:
        # Nếu chỉ muốn xem trực tiếp, `render_mode` là "human".
        render_mode = "human"
    elif args.record:
        # Nếu chỉ muốn ghi lại (không xem trực tiếp), `render_mode` là "rgb_array".
        render_mode = "rgb_array"
    # Nếu không render và không record, `render_mode` là None (mặc định).

    # Tạo đối tượng môi trường PacmanEnv.
    env = PacmanEnv(env_name=env_name, render_mode=render_mode, stack_frames=args.stack_frames)

    # Lấy thông tin về không gian state và action từ môi trường.
    state_shape = env.observation_space.shape  # Ví dụ: (4, 84, 84)
    n_actions = env.action_space.n           # Số lượng hành động khả thi.
    print(f"State shape: {state_shape}, Action space: {n_actions}")

    # Tạo đối tượng agent.
    # Lưu ý: Khi test, nhiều tham số của agent (như learning_rate, epsilon_decay)
    # không quan trọng vì agent không huấn luyện. Tuy nhiên, chúng vẫn cần được cung cấp
    # để khởi tạo đối tượng agent. Epsilon nên được đặt ở mức thấp (ví dụ: 0.01)
    # để agent chủ yếu thực hiện exploitation.
    agent = DoubleDQNAgent(
        state_shape=state_shape,
        n_actions=n_actions,
        device=args.device,
        learning_rate=Config.LEARNING_RATE,  # Lấy từ Config, không quan trọng khi test
        gamma=Config.GAMMA,                  # Lấy từ Config, có thể ảnh hưởng nếu agent vẫn tính toán giá trị
        epsilon_start=0.01,                  # Epsilon thấp để exploitation
        epsilon_final=0.01,                  # Epsilon không thay đổi
        epsilon_decay=1,                     # Decay không có tác dụng
        buffer_size=Config.BUFFER_SIZE,      # Lấy từ Config, không quan trọng khi test
        batch_size=Config.BATCH_SIZE,        # Lấy từ Config, không quan trọng khi test
        target_update=Config.TARGET_UPDATE   # Lấy từ Config, không quan trọng khi test
    )

    # Kiểm tra xem đường dẫn model có được cung cấp không.
    if args.model_path is None:
        # Điều này chỉ xảy ra nếu TestConfig.MODEL_PATH là None VÀ người dùng không cung cấp --model-path.
        raise ValueError("--model-path must be specified if not set in TestConfig")

    # Load trọng số của model đã huấn luyện vào agent.
    print(f"Loading model from {args.model_path}")
    agent.load(args.model_path) # Hàm load này đã được định nghĩa trong lớp DoubleDQNAgent.

    # Chạy quá trình test.
    print(f"Testing agent for {args.episodes} episodes")
    rewards = test(env, agent, args.episodes, render=args.render, record=args.record, save_dir=args.save_dir)

    # In ra các kết quả thống kê.
    print("\nTest Results:")
    print(f"Average reward: {np.mean(rewards):.2f}")
    print(f"Standard deviation: {np.std(rewards):.2f}")
    print(f"Min reward: {np.min(rewards):.2f}")
    print(f"Max reward: {np.max(rewards):.2f}")

    # Vẽ biểu đồ phần thưởng qua các episode.
    plt.figure(figsize=(10, 5)) # Tạo một figure mới với kích thước 10x5 inches.
    plt.plot(rewards, 'b-')     # Vẽ đường biểu diễn phần thưởng (màu xanh lam, nét liền).
    # Vẽ một đường ngang thể hiện phần thưởng trung bình.
    plt.axhline(y=float(np.mean(rewards)), color='r', linestyle='-', label=f"Average: {np.mean(rewards):.2f}")
    plt.xlabel("Episode")       # Đặt nhãn cho trục x.
    plt.ylabel("Reward")        # Đặt nhãn cho trục y.
    plt.title(f"Test Rewards ({args.episodes} episodes)") # Đặt tiêu đề cho biểu đồ.
    plt.legend() # Hiển thị chú giải (label của đường trung bình).

    # Lưu biểu đồ thành file ảnh.
    # Thư mục lưu sẽ là `args.save_dir` nếu có record, ngược lại là thư mục hiện tại.
    result_dir = args.save_dir if args.record else "."
    plt.savefig(os.path.join(result_dir, "test_rewards.png"))
    print(f"Saved test rewards plot to {os.path.join(result_dir, 'test_rewards.png')}")

    # Đóng môi trường để giải phóng tài nguyên.
    env.close()


# Điểm vào của script: nếu script được chạy trực tiếp (không phải import), hàm main() sẽ được gọi.
if __name__ == "__main__":
    main()