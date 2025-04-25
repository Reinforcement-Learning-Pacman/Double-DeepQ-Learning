# config.py
import ale_py
"""
Cấu hình cho dự án Pacman Double DQN
"""

class Config:
    # Cấu hình môi trường
    ENV_NAME = "ale_py:ALE/MsPacman-v5"  # Thay đổi từ "Pacman-v0"
    STACK_FRAMES = 4                    # Số frame để stack
    FRAME_SIZE = (84, 84)               # Kích thước frame sau khi resize
    
    # Cấu hình agent
    LEARNING_RATE = 2.5e-4              # Tăng tốc độ học
    GAMMA = 0.99                        # Hệ số discount
    EPSILON_START = 1.0                 # Epsilon ban đầu
    EPSILON_FINAL = 0.1                 # Tăng epsilon cuối cùng
    EPSILON_DECAY = 50000               # Giảm tốc độ decay (huấn luyện nhanh hơn)
    
    # Cấu hình replay buffer
    BUFFER_SIZE = 50000                 # Giảm kích thước buffer
    BATCH_SIZE = 32                     # Kích thước batch
    
    # Cấu hình huấn luyện
    TARGET_UPDATE = 500                 # Giảm số bước giữa mỗi lần cập nhật target network
    TRAIN_STEPS = 500000                # Giảm tổng số bước huấn luyện
    EVAL_INTERVAL = 5000                # Giảm khoảng cách giữa các lần đánh giá
    EVAL_EPISODES = 5                   # Số episode để đánh giá
    
    # Cấu hình lưu trữ
    SAVE_DIR = "checkpoints"            # Thư mục lưu model
    LOG_DIR = "logs"                    # Thư mục log
    SEED = 42                           # Seed cho random

class TestConfig:
    # Cấu hình kiểm thử
    MODEL_PATH = None                   # Đường dẫn đến model đã huấn luyện
    TEST_EPISODES = 10                  # Số episode để kiểm thử
    RENDER = True                       # Bật render môi trường
    RECORD = False                      # Có ghi lại video không
    VIDEO_DIR = "videos"                # Thư mục lưu video