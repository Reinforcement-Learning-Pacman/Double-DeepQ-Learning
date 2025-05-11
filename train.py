import argparse  # Thư viện để phân tích các đối số dòng lệnh.
import os        # Thư viện cung cấp các hàm để tương tác với hệ điều hành.
import numpy as np # NumPy cho các phép toán số học.
import torch     # PyTorch cho deep learning.
from torch.utils.tensorboard import SummaryWriter # SummaryWriter để ghi log cho TensorBoard.
from tqdm import tqdm  # tqdm để hiển thị thanh tiến trình (progress bar).
from collections import deque # deque để tạo một cửa sổ trượt cho việc tính phần thưởng trung bình.

from env import PacmanEnv  # Import lớp PacmanEnv đã được định nghĩa.
from agent import DoubleDQNAgent  # Import lớp DoubleDQNAgent đã được định nghĩa.
from utils import set_seed, create_dirs, get_timestamp, plot_rewards # Import các hàm tiện ích.
from configs import Config # Import lớp Config chung từ file configs.py


def parse_args() -> argparse.Namespace:
    """Parse các tham số dòng lệnh cho script train."""
    parser = argparse.ArgumentParser(description="Train Double DQN for Pacman")

    # --- Các tham số liên quan đến Môi trường ---
    parser.add_argument("--env", type=str, default=Config.ENV_NAME,
                        help="Pacman environment name (e.g., ALE/Pacman-v5)")
    parser.add_argument("--stack-frames", type=int, default=Config.STACK_FRAMES,
                        help="Number of frames to stack as state")

    # --- Các siêu tham số (Hyperparameters) cho việc học ---
    parser.add_argument("--lr", type=float, default=Config.LEARNING_RATE,
                        help="Learning rate for the optimizer")
    parser.add_argument("--gamma", type=float, default=Config.GAMMA,
                        help="Discount factor for future rewards")
    parser.add_argument("--eps-start", type=float, default=Config.EPSILON_START,
                        help="Starting value of epsilon for epsilon-greedy exploration")
    parser.add_argument("--eps-final", type=float, default=Config.EPSILON_FINAL,
                        help="Final value of epsilon after decay")
    parser.add_argument("--eps-decay", type=int, default=Config.EPSILON_DECAY,
                        help="Number of steps over which epsilon decays")

    # --- Các tham số liên quan đến Quá trình Huấn luyện ---
    parser.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE,
                        help="Batch size for sampling from the replay buffer")
    parser.add_argument("--buffer-size", type=int, default=Config.BUFFER_SIZE,
                        help="Maximum size of the replay buffer")
    parser.add_argument("--target-update", type=int, default=Config.TARGET_UPDATE,
                        help="Frequency (in steps) of updating the target network")
    parser.add_argument("--train-steps", type=int, default=Config.TRAIN_STEPS,
                        help="Total number of training steps (agent-environment interactions)")
    parser.add_argument("--eval-interval", type=int, default=Config.EVAL_INTERVAL,
                        help="Frequency (in steps) of evaluating the agent")
    parser.add_argument("--eval-episodes", type=int, default=Config.EVAL_EPISODES,
                        help="Number of episodes to run for each evaluation")

    # --- Các tham số khác (Lưu trữ, Thiết bị, Seed) ---
    parser.add_argument("--save-dir", type=str, default=Config.SAVE_DIR,
                        help="Base directory to save trained models")
    parser.add_argument("--log-dir", type=str, default=Config.LOG_DIR,
                        help="Base directory for TensorBoard logs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for PyTorch computations (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=Config.SEED,
                        help="Random seed for reproducibility")

    return parser.parse_args() # Phân tích và trả về các đối số.


def evaluate(env: PacmanEnv, agent: DoubleDQNAgent, num_episodes: int) -> float:
    """
    Đánh giá hiệu suất của agent trên một số episode nhất định.
    Trong quá trình đánh giá, agent sẽ không học (không cập nhật mạng, không exploration).

    Args:
        env (PacmanEnv): Môi trường để đánh giá (nên là một instance riêng biệt với môi trường huấn luyện).
        agent (DoubleDQNAgent): Agent cần được đánh giá.
        num_episodes (int): Số lượng episode để chạy đánh giá.

    Returns:
        float: Phần thưởng trung bình đạt được qua các episode đánh giá.
    """
    total_rewards = []  # Danh sách để lưu tổng phần thưởng của mỗi episode đánh giá.

    for _ in range(num_episodes):
        state, _ = env.reset()  # Reset môi trường.
        done = False
        truncated = False
        total_reward = 0
        # Giới hạn số bước tối đa cho mỗi episode đánh giá để tránh kẹt vô hạn.
        max_steps_eval = 50000  # Tăng giới hạn này nếu các episode Pacman thường kéo dài hơn.
        steps = 0

        # Chơi một episode.
        while not (done or truncated) and steps < max_steps_eval:
            # Agent chọn hành động mà không exploration (training=False).
            action = agent.select_action(state, training=False)
            next_state, reward, done, truncated, _ = env.step(action)

            total_reward += reward
            state = next_state
            steps += 1

        total_rewards.append(total_reward)

    average_reward = np.mean(total_rewards) # Tính phần thưởng trung bình.
    return float(average_reward)  # Ép kiểu sang float chuẩn của Python.


def main() -> None:
    """Hàm chính để chạy quá trình huấn luyện."""
    # Phân tích các đối số dòng lệnh.
    args = parse_args()

    # Đặt seed ngẫu nhiên để đảm bảo kết quả có thể tái tạo.
    set_seed(args.seed)

    # Tạo các thư mục để lưu model và log TensorBoard.
    # Sử dụng timestamp để mỗi lần chạy huấn luyện sẽ có thư mục riêng.
    timestamp = get_timestamp() # Lấy timestamp hiện tại (ví dụ: "20231027_103045").
    save_dir = os.path.join(args.save_dir, timestamp) # Thư mục lưu model, ví dụ: "saved_models/20231027_103045".
    log_dir = os.path.join(args.log_dir, timestamp)   # Thư mục lưu log, ví dụ: "logs/20231027_103045".
    create_dirs([save_dir, log_dir]) # Tạo các thư mục này nếu chúng chưa tồn tại.

    # Thiết lập TensorBoard SummaryWriter.
    # Tất cả log sẽ được ghi vào thư mục `log_dir`.
    writer = SummaryWriter(log_dir=log_dir)

    # Tạo môi trường huấn luyện và môi trường đánh giá.
    # Nên sử dụng các instance môi trường riêng biệt cho huấn luyện và đánh giá
    # để tránh ảnh hưởng lẫn nhau (ví dụ: trạng thái ngẫu nhiên).
    env = PacmanEnv(env_name=args.env, stack_frames=args.stack_frames)
    eval_env = PacmanEnv(env_name=args.env, stack_frames=args.stack_frames) # Môi trường riêng cho đánh giá.

    # Lấy thông tin về không gian state và action.
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n
    print(f"State shape: {state_shape}, Action space: {n_actions}")

    # Tạo đối tượng agent.
    agent = DoubleDQNAgent(
        state_shape=state_shape,
        n_actions=n_actions,
        device=args.device,
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon_start=args.eps_start,
        epsilon_final=args.eps_final,
        epsilon_decay=args.eps_decay,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update=args.target_update
    )

    # Lưu thông tin cấu hình của lần chạy này vào một file text trong thư mục log.
    config_file = os.path.join(log_dir, "config.txt")
    with open(config_file, "w") as f:
        for arg_name, arg_value in vars(args).items(): # `vars(args)` trả về dict của các thuộc tính của args.
            f.write(f"{arg_name}: {arg_value}\n")

    # --- Vòng lặp Huấn luyện ---
    state, _ = env.reset()  # Reset môi trường để lấy state ban đầu.
    episode_reward = 0      # Tổng phần thưởng của episode hiện tại.
    episode_rewards = []    # Danh sách lưu tổng phần thưởng của tất cả các episode đã hoàn thành.
    avg_rewards = []        # Danh sách lưu phần thưởng trung bình (trượt) qua các episode.
    # `deque` với `maxlen=100` tạo một cửa sổ trượt lưu 100 phần thưởng episode gần nhất.
    rewards_window = deque(maxlen=100)

    episode = 0  # Biến đếm số episode đã hoàn thành.
    best_eval_reward = float('-inf')  # Lưu trữ phần thưởng đánh giá tốt nhất để lưu model "best".

    # Tạo thanh tiến trình `tqdm` cho vòng lặp huấn luyện.
    # Vòng lặp chạy từ 1 đến `args.train_steps`.
    progress_bar = tqdm(range(1, args.train_steps + 1), desc="Training", unit="step")

    for step in progress_bar:
        # Agent chọn hành động dựa trên state hiện tại (sử dụng epsilon-greedy).
        action = agent.select_action(state) # `training=True` là mặc định trong `agent.select_action`.
        # Thực hiện hành động trong môi trường.
        next_state, reward, done, truncated, _ = env.step(action)

        # Lưu transition (kinh nghiệm) vào replay buffer của agent.
        # `done or truncated` để xử lý cả hai trường hợp kết thúc episode.
        agent.buffer.add(state, action, reward, next_state, done or truncated)

        # Cập nhật tổng phần thưởng của episode hiện tại.
        episode_reward += reward

        # Chuyển sang state tiếp theo.
        state = next_state

        # Huấn luyện agent (thực hiện một bước tối ưu hóa).
        # `agent.optimize()` sẽ lấy mẫu từ buffer, tính loss và cập nhật policy network.
        loss = agent.optimize()

        # Cập nhật target network định kỳ.
        if step % args.target_update == 0:
            agent.update_target_network()

        # --- Ghi log các thông số huấn luyện cho TensorBoard ---
        if loss is not None: # Loss có thể là None nếu buffer chưa đủ lớn.
            writer.add_scalar("train/loss", loss, step) # Ghi giá trị loss tại bước `step`.

        writer.add_scalar("train/epsilon", agent.epsilon, step) # Ghi giá trị epsilon hiện tại.

        # Xử lý khi một episode kết thúc.
        if done or truncated:
            writer.add_scalar("train/episode_reward", episode_reward, episode) # Ghi phần thưởng của episode vừa kết thúc.
            episode_rewards.append(episode_reward)
            rewards_window.append(episode_reward) # Thêm vào cửa sổ trượt.

            if len(rewards_window) > 0: # Đảm bảo deque không rỗng.
                avg_reward = np.mean(rewards_window) # Tính phần thưởng trung bình của 100 episode gần nhất.
                avg_rewards.append(avg_reward)
                writer.add_scalar("train/avg_reward_100_episodes", avg_reward, episode) # Ghi phần thưởng trung bình.

                # Cập nhật thông tin hiển thị trên thanh tiến trình `tqdm`.
                progress_bar.set_postfix({
                    "episode": episode,
                    "reward": f"{episode_reward:.1f}", # Phần thưởng của episode vừa xong.
                    "avg_100": f"{avg_reward:.1f}"     # Phần thưởng trung bình 100 ep gần nhất.
                })

            # Reset môi trường để bắt đầu episode mới.
            state, _ = env.reset()
            episode_reward = 0 # Reset phần thưởng cho episode mới.
            episode += 1       # Tăng biến đếm episode.

        # --- Đánh giá định kỳ ---
        if step % args.eval_interval == 0 and step > 0: # Bắt đầu đánh giá sau một số bước nhất định.
            # Gọi hàm `evaluate` để lấy phần thưởng trung bình trên môi trường đánh giá.
            eval_reward = evaluate(eval_env, agent, args.eval_episodes)
            writer.add_scalar("eval/average_reward", eval_reward, step) # Ghi phần thưởng đánh giá.

            print(f"\nEvaluation at step {step}: Average reward = {eval_reward:.2f}")

            # Lưu model nếu nó đạt được phần thưởng đánh giá tốt hơn.
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                agent.save(os.path.join(save_dir, "model_best.pth")) # Lưu model tốt nhất.
                # Chú ý: Tên file có thể là .pth hoặc .pt
                print(f"New best model saved with reward {best_eval_reward:.2f}")

            # Lưu checkpoint định kỳ (bất kể có phải là best model hay không).
            agent.save(os.path.join(save_dir, f"model_step_{step}.pth"))

            # Vẽ và lưu biểu đồ phần thưởng huấn luyện.
            if len(episode_rewards) > 0:
                plot_rewards(
                    episode_rewards,    # Danh sách phần thưởng của từng episode.
                    avg_rewards,        # Danh sách phần thưởng trung bình trượt.
                    window_size=100,    # Kích thước cửa sổ trượt đã dùng.
                    filename=os.path.join(log_dir, f"rewards_step_{step}.png") # Tên file lưu biểu đồ.
                )

    # --- Kết thúc quá trình huấn luyện ---
    # Lưu model cuối cùng sau khi hoàn tất tất cả các bước huấn luyện.
    agent.save(os.path.join(save_dir, "model_final.pth"))

    # Vẽ và lưu biểu đồ phần thưởng cuối cùng.
    if len(episode_rewards) > 0:
        plot_rewards(
            episode_rewards,
            avg_rewards,
            window_size=100,
            filename=os.path.join(log_dir, "rewards_final.png")
        )

    # Đóng các môi trường.
    env.close()
    eval_env.close()

    # Đóng TensorBoard writer.
    writer.close()

    print(f"\nTraining completed. Best evaluation reward: {best_eval_reward:.2f}")
    print(f"Models saved in {save_dir}")
    print(f"Logs saved in {log_dir}")


# Điểm vào của script: nếu script được chạy trực tiếp, hàm main() sẽ được gọi.
if __name__ == "__main__":
    main()