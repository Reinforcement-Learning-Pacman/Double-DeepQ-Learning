# config.py
import ale_py
"""
Cấu hình cho dự án Pacman Double DQN
"""

class Config:
    #Config env
    ENV_NAME = "ale_py:ALE/MsPacman-v5"  
    STACK_FRAMES = 4                    
    FRAME_SIZE = (84, 84)             
    
    #COnfig agent
    LEARNING_RATE = 5e-5              
    GAMMA = 0.99                        
    EPSILON_START = 1.0                 
    EPSILON_FINAL = 0.01                 
    EPSILON_DECAY = 220000               
    
    #Config buffer
    BUFFER_SIZE = 100000                 
    BATCH_SIZE = 32                     
    
    #Conf Train
    TARGET_UPDATE = 10000                
    TRAIN_STEPS = 1000000                
    EVAL_INTERVAL = 50000                
    EVAL_EPISODES = 5                   
        
    #save
    SAVE_DIR = "checkpoints"          
    LOG_DIR = "logs"                 
    SEED = 42                         

class TestConfig:
    MODEL_PATH = None                  
    TEST_EPISODES = 10                  
    RENDER = True                    
    RECORD = False                    
    VIDEO_DIR = "videos"            
