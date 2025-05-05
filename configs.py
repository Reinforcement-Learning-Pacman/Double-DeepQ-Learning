# config.py
import ale_py
"""
Cấu hình cho dự án Pacman Double DQN
"""

class Config:
    # Cấu hình môi trường
    ENV_NAME = "ale_py:ALE/MsPacman-v5"  
    STACK_FRAMES = 4                    
    FRAME_SIZE = (84, 84)             
    
    # Cấu hình agent
    LEARNING_RATE = 2.5e-4              # Tăng tốc độ học
    GAMMA = 0.99                        # Hệ số discount
    EPSILON_START = 1.0                 # Epsilon ban đầu
    EPSILON_FINAL = 0.1                 # Tăng epsilon cuối cùng
    EPSILON_DECAY = 20000               # Giảm tốc độ decay (huấn luyện nhanh hơn)
    
    # Cấu hình replay buffer
    BUFFER_SIZE = 10000                 # Giảm kích thước buffer
    BATCH_SIZE = 32                     # Kích thước batch
    
    # Cấu hình huấn luyện
    TARGET_UPDATE = 1000                 # Giảm số bước giữa mỗi lần cập nhật target network
    TRAIN_STEPS = 100000                # Giảm tổng số bước huấn luyện
    EVAL_INTERVAL = 10000                # Giảm khoảng cách giữa các lần đánh giá
    EVAL_EPISODES = 3                   # Số episode để đánh giá
        
    # Cấu hình lưu trữ
    SAVE_DIR = "checkpoints"          
    LOG_DIR = "logs"                 
    SEED = 42                         

class TestConfig:
    # Cấu hình kiểm thử
    MODEL_PATH = None                  
    TEST_EPISODES = 10                  
    RENDER = True                    
    RECORD = False                    
    VIDEO_DIR = "videos"            
