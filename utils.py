import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import random
from datetime import datetime
from typing import List


def set_seed(seed):
    """Set seed for all random sources to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dirs(dirs: List[str]) -> None:
    """Tạo các thư mục nếu chưa tồn tại"""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def get_timestamp() -> str:
    """Lấy timestamp hiện tại"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def plot_rewards(episode_rewards: List[float], avg_rewards: List[float], window_size: int = 100,
                 filename: str = 'rewards_plot.png') -> None:
    """Vẽ đồ thị phần thưởng"""
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, alpha=0.5, label='Episode Reward')
    plt.plot(avg_rewards, label=f'Average Reward (window={window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Training Rewards over Episodes')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def display_frames_as_gif(frames: List[np.ndarray], filename: str = 'game.gif', fps: int = 30) -> None:
    """Lưu một chuỗi frame dưới dạng GIF"""
    from PIL import Image

    print(f"Attempting to save {len(frames)} frames to {filename}")

    if len(frames) == 0:
        print("Error: No frames to save!")
        return

    # Đảm bảo thư mục tồn tại
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)

    try:
        # Đảm bảo frames có định dạng đúng (uint8)
        frames_np = [np.array(frame).astype(np.uint8) if frame is not None else np.zeros((84, 84, 3), dtype=np.uint8)
                     for frame in frames]
        frames_pil = [Image.fromarray(frame) for frame in frames_np]

        # Lưu GIF
        frames_pil[0].save(
            filename,
            save_all=True,
            append_images=frames_pil[1:],
            duration=1000 / fps,
            loop=0
        )
        print(f"Successfully saved GIF to {filename}")
    except Exception as e:
        print(f"Error saving GIF: {e}")
        import traceback
        traceback.print_exc()