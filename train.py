# train.py
# Comment: Tên file là "train.py". File này chứa logic chính cho việc huấn luyện agent.

import argparse  # Thư viện chuẩn của Python để phân tích các đối số dòng lệnh (command-line arguments).
import os        # Thư viện chuẩn của Python cung cấp các hàm để tương tác với hệ điều hành (ví dụ: tạo thư mục, nối đường dẫn).
import numpy as np # Thư viện NumPy, dùng cho các phép toán số học hiệu quả, đặc biệt là với mảng và ma trận.
import torch       # Thư viện PyTorch, một framework deep learning phổ biến.
from torch.utils.tensorboard import SummaryWriter # Lớp từ PyTorch để ghi dữ liệu cho TensorBoard, một công cụ trực quan hóa.
from tqdm import tqdm  # Thư viện để tạo thanh tiến trình (progress bar) cho các vòng lặp, giúp theo dõi tiến độ.
from collections import deque # Lớp `deque` (double-ended queue) từ thư viện `collections`, hữu ích để tạo cửa sổ trượt.

from env import PacmanEnv  # Import lớp `PacmanEnv` từ file `env.py` (định nghĩa môi trường game Pacman đã được tùy chỉnh).
from agent import DoubleDQNAgent  # Import lớp `DoubleDQNAgent` từ file `agent.py` (định nghĩa agent học tăng cường).
from utils import set_seed, create_dirs, get_timestamp, plot_rewards # Import các hàm tiện ích từ file `utils.py`.
from configs import Config # Import lớp `Config` từ file `configs.py` (chứa các cấu hình mặc định cho việc huấn luyện).


def parse_args() -> argparse.Namespace:
    """
    Phân tích các tham số (arguments) được truyền vào từ dòng lệnh khi chạy script.
    Hàm này định nghĩa các cờ (flags) và tùy chọn mà người dùng có thể sử dụng.
    Trả về một đối tượng `argparse.Namespace` chứa các giá trị của tham số đã được parse.
    """
    # Tạo một đối tượng `ArgumentParser`. `description` là mô tả ngắn về chương trình,
    # sẽ hiển thị khi người dùng yêu cầu trợ giúp (ví dụ: `python train.py --help`).
    parser = argparse.ArgumentParser(description="Train Double DQN for Pacman")

    # --- Định nghĩa các nhóm tham số ---

    # Tham số liên quan đến Môi trường (Environment)
    # Thêm tham số `--env` để chỉ định tên môi trường game.
    # `type=str`: Kiểu dữ liệu của tham số là chuỗi.
    # `default=Config.ENV_NAME`: Giá trị mặc định lấy từ lớp `Config`.
    # `help`: Chuỗi mô tả ý nghĩa của tham số.
    parser.add_argument("--env", type=str, default=Config.ENV_NAME,
                        help="Pacman environment name")
    # Thêm tham số `--stack-frames` để chỉ định số lượng frame ảnh được xếp chồng làm trạng thái.
    parser.add_argument("--stack-frames", type=int, default=Config.STACK_FRAMES,
                        help="Number of frames to stack")

    # Tham số liên quan đến Hyperparameters (Siêu tham số của thuật toán học)
    # Thêm tham số `--lr` (learning rate - tốc độ học).
    parser.add_argument("--lr", type=float, default=Config.LEARNING_RATE,
                        help="Learning rate")
    # Thêm tham số `--gamma` (discount factor - hệ số chiết khấu).
    parser.add_argument("--gamma", type=float, default=Config.GAMMA,
                        help="Discount factor")
    # Thêm tham số `--eps-start` (epsilon start - giá trị epsilon ban đầu cho epsilon-greedy).
    parser.add_argument("--eps-start", type=float, default=Config.EPSILON_START,
                        help="Starting value of epsilon")
    # Thêm tham số `--eps-final` (epsilon final - giá trị epsilon cuối cùng).
    parser.add_argument("--eps-final", type=float, default=Config.EPSILON_FINAL,
                        help="Final value of epsilon")
    # Thêm tham số `--eps-decay` (epsilon decay - số bước để epsilon giảm từ start xuống final).
    parser.add_argument("--eps-decay", type=int, default=Config.EPSILON_DECAY,
                        help="Number of steps for epsilon decay")

    # Tham số liên quan đến Training parameters (Các tham số của quá trình huấn luyện)
    # Thêm tham số `--batch-size` (kích thước batch).
    parser.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE,
                        help="Batch size for training")
    # Thêm tham số `--buffer-size` (kích thước replay buffer).
    parser.add_argument("--buffer-size", type=int, default=Config.BUFFER_SIZE,
                        help="Size of replay buffer")
    # Thêm tham số `--target-update` (tần suất cập nhật target network).
    parser.add_argument("--target-update", type=int, default=Config.TARGET_UPDATE,
                        help="Number of steps between target network updates")
    # Thêm tham số `--train-steps` (tổng số bước huấn luyện).
    parser.add_argument("--train-steps", type=int, default=Config.TRAIN_STEPS,
                        help="Total number of training steps")
    # Thêm tham số `--eval-interval` (tần suất đánh giá agent).
    parser.add_argument("--eval-interval", type=int, default=Config.EVAL_INTERVAL,
                        help="Interval between evaluations")
    # Thêm tham số `--eval-episodes` (số episode để chạy khi đánh giá).
    parser.add_argument("--eval-episodes", type=int, default=Config.EVAL_EPISODES,
                        help="Number of episodes for evaluation")

    # Tham số Misc (Linh tinh)
    # Thêm tham số `--save-dir` (thư mục lưu model).
    parser.add_argument("--save-dir", type=str, default=Config.SAVE_DIR,
                        help="Directory to save models")
    # Thêm tham số `--log-dir` (thư mục lưu log TensorBoard).
    parser.add_argument("--log-dir", type=str, default=Config.LOG_DIR,
                        help="Directory for tensorboard logs")
    # Thêm tham số `--device` (thiết bị tính toán: "cuda" hoặc "cpu").
    # Mặc định kiểm tra nếu có GPU (CUDA) thì dùng "cuda", nếu không thì dùng "cpu".
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training")
    # Thêm tham số `--seed` (hạt giống ngẫu nhiên).
    parser.add_argument("--seed", type=int, default=Config.SEED,
                        help="Random seed")

    # Phân tích các đối số được truyền vào từ dòng lệnh (sử dụng `sys.argv[1:]` ngầm định)
    # và trả về một đối tượng `argparse.Namespace` chứa các giá trị của các tham số.
    return parser.parse_args()


def evaluate(env: PacmanEnv, agent: DoubleDQNAgent, num_episodes: int) -> float:
    """
    Đánh giá hiệu suất của agent trên một số episodes.
    Hàm này chạy agent trong môi trường đánh giá mà không cập nhật trọng số của nó.
    Args:
        env: Đối tượng môi trường (PacmanEnv) để thực hiện đánh giá.
        agent: Đối tượng agent (DoubleDQNAgent) cần được đánh giá.
        num_episodes: Số lượng episode sẽ được chạy để tính toán phần thưởng trung bình.
    Returns:
        Phần thưởng trung bình (float) mà agent đạt được qua `num_episodes`.
    """
    total_rewards = []  # Danh sách để lưu trữ tổng phần thưởng của mỗi episode đánh giá.

    # Chạy `num_episodes` lần.
    for _ in range(num_episodes):
        state, _ = env.reset()  # Reset môi trường để bắt đầu một episode mới, nhận trạng thái ban đầu.
        done = False            # Cờ báo hiệu episode kết thúc do agent thua/thắng.
        truncated = False       # Cờ báo hiệu episode kết thúc do vượt quá giới hạn thời gian của môi trường (ít dùng ở đây).
        total_reward = 0      # Khởi tạo tổng phần thưởng cho episode hiện tại.
        max_steps = 50000     # Đặt một giới hạn số bước tối đa cho mỗi episode đánh giá
                              # để tránh trường hợp agent bị kẹt trong vòng lặp vô hạn.
        steps = 0             # Biến đếm số bước trong episode hiện tại.

        # Vòng lặp chính của một episode: tiếp tục cho đến khi `done` hoặc `truncated` hoặc `steps` đạt `max_steps`.
        while not (done or truncated) and steps < max_steps:
            # Agent chọn hành động dựa trên trạng thái hiện tại.
            # `training=False` được truyền vào để agent sử dụng chính sách tham lam (greedy policy),
            # tức là chọn hành động tốt nhất theo Q-values mà không có exploration (epsilon = 0 hoặc rất nhỏ).
            action = agent.select_action(state, training=False)
            # Thực hiện hành động đã chọn trong môi trường.
            # Nhận về trạng thái kế tiếp (`next_state`), phần thưởng (`reward`),
            # cờ kết thúc (`done`, `truncated`), và thông tin bổ sung (`_`, không dùng ở đây).
            next_state, reward, done, truncated, _ = env.step(action)

            total_reward += reward  # Cộng dồn phần thưởng nhận được.
            state = next_state      # Cập nhật trạng thái hiện tại thành trạng thái kế tiếp.
            steps += 1              # Tăng biến đếm số bước.

        total_rewards.append(total_reward) # Sau khi episode kết thúc, thêm tổng phần thưởng vào danh sách.

    average_reward = np.mean(total_rewards) # Tính phần thưởng trung bình của tất cả các episode đánh giá.
    return float(average_reward)  # Ép kiểu kết quả từ `numpy.float64` sang `float` chuẩn của Python.


def main() -> None:
    """
    Hàm chính của script, điều khiển toàn bộ quá trình huấn luyện agent.
    """
    # Bước 1: Parse các tham số dòng lệnh.
    args = parse_args()

    # Bước 2: Đặt seed ngẫu nhiên cho Python, NumPy, và PyTorch.
    # Điều này giúp đảm bảo kết quả huấn luyện có thể lặp lại được (reproducible)
    # nếu chạy lại với cùng các tham số và cùng seed.
    set_seed(args.seed)

    # Bước 3: Tạo các thư mục để lưu trữ model và log.
    timestamp = get_timestamp() # Lấy một chuỗi timestamp hiện tại (ví dụ: "20231027_103000").
                                # Sử dụng timestamp giúp mỗi lần chạy huấn luyện có một thư mục lưu trữ riêng biệt,
                                # tránh ghi đè lên kết quả của các lần chạy trước.
    save_dir = os.path.join(args.save_dir, timestamp) # Tạo đường dẫn đến thư mục lưu model (ví dụ: "checkpoints/20231027_103000").
    log_dir = os.path.join(args.log_dir, timestamp)   # Tạo đường dẫn đến thư mục lưu log TensorBoard (ví dụ: "logs/20231027_103000").
    create_dirs([save_dir, log_dir]) # Gọi hàm tiện ích để tạo các thư mục này nếu chúng chưa tồn tại.

    # Bước 4: Thiết lập TensorBoard SummaryWriter.
    # `SummaryWriter` được dùng để ghi các dữ liệu (ví dụ: loss, reward, epsilon) trong quá trình huấn luyện,
    # sau đó có thể được trực quan hóa bằng TensorBoard.
    writer = SummaryWriter(log_dir=log_dir) # Chỉ định thư mục log cho writer.

    # Bước 5: Tạo môi trường huấn luyện (`env`) và môi trường đánh giá (`eval_env`).
    # Sử dụng hai đối tượng môi trường riêng biệt:
    # - `env`: Dùng cho việc thu thập kinh nghiệm và huấn luyện agent.
    # - `eval_env`: Dùng riêng cho việc đánh giá định kỳ, giúp kết quả đánh giá khách quan hơn,
    #   không bị ảnh hưởng bởi trạng thái ngẫu nhiên hoặc các yếu tố khác của môi trường huấn luyện.
    env = PacmanEnv(env_name=args.env, stack_frames=args.stack_frames)
    eval_env = PacmanEnv(env_name=args.env, stack_frames=args.stack_frames)

    # Bước 6: Lấy thông tin về không gian trạng thái và không gian hành động từ môi trường.
    state_shape = env.observation_space.shape # Hình dạng của một trạng thái (ví dụ: (4, 84, 84) cho 4 frame ảnh 84x84).
    n_actions = env.action_space.n          # Số lượng hành động mà agent có thể thực hiện (ví dụ: 5 cho Pacman).
    print(f"State shape: {state_shape}, Action space: {n_actions}") # In thông tin này ra console.

    # Bước 7: Tạo đối tượng agent (DoubleDQNAgent).
    # Truyền các tham số đã parse từ dòng lệnh (hoặc giá trị mặc định) vào constructor của agent.
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

    # Bước 8: Lưu thông tin cấu hình của lần chạy huấn luyện này vào một file text.
    # Điều này hữu ích để sau này có thể xem lại các tham số đã được sử dụng cho một lần chạy cụ thể.
    config_file = os.path.join(log_dir, "config.txt") # Tạo đường dẫn đến file config.txt trong thư mục log.
    with open(config_file, "w") as f: # Mở file ở chế độ ghi ("w").
        for arg_name in vars(args): # `vars(args)` trả về một dictionary chứa tên và giá trị của các tham số đã parse.
            # Ghi tên tham số và giá trị của nó vào file, mỗi tham số trên một dòng.
            f.write(f"{arg_name}: {getattr(args, arg_name)}\n")

    # Bước 9: Khởi tạo các biến cho vòng lặp huấn luyện.
    state, _ = env.reset() # Reset môi trường huấn luyện và nhận trạng thái ban đầu.
    episode_reward = 0     # Biến lưu trữ tổng phần thưởng của episode đang diễn ra.
    episode_rewards = []   # Danh sách lưu trữ tổng phần thưởng của tất cả các episode đã hoàn thành.
    avg_rewards = []       # Danh sách lưu trữ phần thưởng trung bình (ví dụ: của 100 episode gần nhất).
    # `deque(maxlen=100)` tạo một hàng đợi hai đầu với kích thước tối đa là 100.
    # Khi thêm phần tử mới vào deque đã đầy, phần tử ở đầu đối diện sẽ bị loại bỏ.
    # Điều này rất tiện để tính trung bình trượt (moving average) của 100 episode gần nhất.
    rewards_window = deque(maxlen=100)

    episode = 0 # Biến đếm số episode đã hoàn thành.
    best_eval_reward = float('-inf') # Biến lưu trữ phần thưởng đánh giá tốt nhất đạt được, khởi tạo là âm vô cùng.

    # Sử dụng `tqdm` để tạo một thanh tiến trình cho vòng lặp huấn luyện.
    # Vòng lặp sẽ chạy từ 1 đến `args.train_steps` (bao gồm cả `args.train_steps`).
    # `desc="Training"` là mô tả hiển thị trên thanh tiến trình.
    progress_bar = tqdm(range(1, args.train_steps + 1), desc="Training")

    # --- Vòng lặp huấn luyện chính ---
    for step in progress_bar: # `step` là bước huấn luyện hiện tại (tương tác với môi trường).
        # Bước 9.1: Agent chọn hành động.
        # `agent.select_action(state)` sẽ sử dụng chiến lược epsilon-greedy để chọn hành động
        # dựa trên trạng thái `state` hiện tại. Epsilon sẽ giảm dần theo `agent.steps_done`.
        action = agent.select_action(state)
        # Bước 9.2: Thực hiện hành động trong môi trường.
        # Nhận về trạng thái kế tiếp (`next_state`), phần thưởng (`reward`),
        # cờ kết thúc (`done`, `truncated`), và thông tin bổ sung (`_`).
        next_state, reward, done, truncated, _ = env.step(action)

        # Bước 9.3: Lưu transition (kinh nghiệm) vào replay buffer của agent.
        # Transition bao gồm (trạng thái hiện tại, hành động, phần thưởng, trạng thái kế tiếp, cờ kết thúc).
        # `done or truncated` được dùng vì cả hai trường hợp này đều đánh dấu sự kết thúc của một tương tác hợp lệ
        # mà từ đó agent có thể học.
        agent.buffer.add(state, action, reward, next_state, done or truncated)

        # Bước 9.4: Cập nhật phần thưởng của episode hiện tại.
        episode_reward += reward

        # Bước 9.5: Chuyển sang trạng thái mới cho bước tiếp theo.
        state = next_state

        # Bước 9.6: Huấn luyện agent (cập nhật mạng nơ-ron).
        # `agent.optimize()` sẽ lấy một batch các transitions từ replay buffer,
        # tính toán loss, và thực hiện một bước gradient descent để cập nhật policy network.
        # Hàm này có thể trả về `None` nếu buffer chưa có đủ mẫu để tạo batch.
        loss = agent.optimize()

        # Bước 9.7: Cập nhật target network định kỳ.
        # Target network được cập nhật bằng cách sao chép trọng số từ policy network
        # sau một số bước huấn luyện (`args.target_update`) nhất định.
        if step % args.target_update == 0:
            agent.update_target_network()

        # Bước 9.8: Ghi log các chỉ số huấn luyện lên TensorBoard.
        if loss is not None: # Chỉ ghi log loss nếu `agent.optimize()` đã thực hiện và trả về giá trị loss.
            # `writer.add_scalar(tag, scalar_value, global_step)`
            # - `tag`: Tên của biểu đồ trên TensorBoard (ví dụ: "train/loss").
            # - `scalar_value`: Giá trị cần ghi log (ví dụ: `loss`).
            # - `global_step`: Bước (thường là `step` huấn luyện) tương ứng với giá trị này.
            writer.add_scalar("train/loss", loss, step)

        # Ghi log giá trị epsilon hiện tại của agent.
        writer.add_scalar("train/epsilon", agent.epsilon, step)

        # Bước 9.9: Xử lý khi một episode kết thúc (do `done` hoặc `truncated`).
        if done or truncated:
            # Ghi log phần thưởng của episode vừa kết thúc. `episode` là chỉ số của episode đã hoàn thành.
            writer.add_scalar("train/episode_reward", episode_reward, episode)
            episode_rewards.append(episode_reward) # Thêm tổng phần thưởng của episode này vào danh sách.
            rewards_window.append(episode_reward)  # Thêm vào cửa sổ trượt để tính trung bình.

            # Tính và ghi log phần thưởng trung bình của 100 episode gần nhất (nếu có đủ dữ liệu).
            if len(rewards_window) > 0:
                avg_reward = np.mean(rewards_window) # Tính trung bình các phần thưởng trong `rewards_window`.
                avg_rewards.append(avg_reward)       # Lưu lại giá trị trung bình này.
                writer.add_scalar("train/avg_reward_100", avg_reward, episode) # Ghi log lên TensorBoard.

                # Cập nhật thông tin hiển thị trên thanh tiến trình `tqdm`.
                # `set_postfix` cho phép hiển thị các thông tin phụ trợ.
                progress_bar.set_postfix({
                    "episode": episode,                 # Số episode hiện tại.
                    "reward": f"{episode_reward:.1f}",  # Phần thưởng của episode vừa kết thúc.
                    "avg_100": f"{avg_reward:.1f}"      # Phần thưởng trung bình của 100 episode gần nhất.
                })

            # Reset môi trường để bắt đầu một episode mới.
            state, _ = env.reset()
            episode_reward = 0 # Reset tổng phần thưởng của episode.
            episode += 1       # Tăng biến đếm số episode đã hoàn thành.

        # Bước 9.10: Đánh giá agent định kỳ.
        # Thực hiện đánh giá sau mỗi `args.eval_interval` bước huấn luyện.
        if step > 0 and step % args.eval_interval == 0: # Thêm `step > 0` để tránh đánh giá ở bước 0.
            # Gọi hàm `evaluate` để chạy agent trên môi trường đánh giá (`eval_env`)
            # trong `args.eval_episodes` lượt chơi và lấy phần thưởng trung bình.
            eval_reward = evaluate(eval_env, agent, args.eval_episodes)
            # Ghi log phần thưởng đánh giá trung bình lên TensorBoard.
            writer.add_scalar("eval/avg_reward", eval_reward, step)

            # In kết quả đánh giá ra console.
            print(f"\nEvaluation at step {step}: Average reward = {eval_reward:.2f}")

            # Lưu model nếu nó đạt được kết quả đánh giá tốt hơn so với `best_eval_reward` trước đó.
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward # Cập nhật `best_eval_reward`.
                # Gọi hàm `save` của agent để lưu trọng số model.
                agent.save(os.path.join(save_dir, "model_best.pt")) # Lưu model tốt nhất với tên "model_best.pt".
                print(f"New best model saved with reward {best_eval_reward:.2f}")

            # Lưu checkpoint (model tại một bước cụ thể) định kỳ, bất kể có phải là tốt nhất hay không.
            # Điều này hữu ích để có thể quay lại một điểm huấn luyện cụ thể nếu cần.
            agent.save(os.path.join(save_dir, f"model_step_{step}.pt")) # Ví dụ: "model_step_10000.pt".

            # Vẽ và lưu biểu đồ phần thưởng (bao gồm phần thưởng từng episode và trung bình trượt).
            if len(episode_rewards) > 0: # Chỉ vẽ nếu có dữ liệu phần thưởng.
                plot_rewards(
                    episode_rewards, # Danh sách các phần thưởng của từng episode.
                    avg_rewards,     # Danh sách các phần thưởng trung bình trượt.
                    window_size=100, # Kích thước cửa sổ trượt (dùng cho chú thích trên đồ thị).
                    # Tên file để lưu đồ thị, bao gồm cả `step` hiện tại.
                    filename=os.path.join(log_dir, f"rewards_step_{step}.png")
                )

    # --- Kết thúc vòng lặp huấn luyện ---

    # Bước 10: Lưu model cuối cùng sau khi hoàn thành tất cả các bước huấn luyện.
    agent.save(os.path.join(save_dir, "model_final.pt"))

    # Bước 11: Vẽ và lưu biểu đồ phần thưởng cuối cùng.
    if len(episode_rewards) > 0:
        plot_rewards(
            episode_rewards,
            avg_rewards,
            window_size=100,
            filename=os.path.join(log_dir, "rewards_final.png")
        )

    # Bước 12: Đóng các môi trường để giải phóng tài nguyên.
    env.close()
    eval_env.close()

    # Bước 13: Đóng TensorBoard SummaryWriter.
    writer.close()

    # Bước 14: In thông báo kết thúc huấn luyện ra console.
    print(f"\nTraining completed. Best evaluation reward: {best_eval_reward:.2f}")
    print(f"Models saved in {save_dir}") # Thông báo vị trí lưu model.
    print(f"Logs saved in {log_dir}")   # Thông báo vị trí lưu log.


if __name__ == "__main__":
    # Đây là điểm vào (entry point) chuẩn của một script Python.
    # Khối code bên trong `if` này chỉ được thực thi khi script được chạy trực tiếp
    # (ví dụ: `python train.py --lr 0.001`), không phải khi được import như một module.
    main()