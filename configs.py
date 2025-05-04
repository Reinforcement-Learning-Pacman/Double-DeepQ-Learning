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
    LEARNING_RATE = 2e-4            
    GAMMA = 0.99                        
    EPSILON_START = 1.0                
    EPSILON_FINAL = 0.1                
    EPSILON_DECAY = 45000            
    # Cấu hình replay buffer
    BUFFER_SIZE = 50000                 
    BATCH_SIZE = 32                     
    
    # Cấu hình huấn luyện
    TARGET_UPDATE = 450               
    TRAIN_STEPS = 500000            
    EVAL_INTERVAL = 5000               
    EVAL_EPISODES = 5                   
    
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
