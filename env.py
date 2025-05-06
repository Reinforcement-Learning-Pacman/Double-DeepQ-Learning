import gymnasium as gym
import numpy as np
import cv2
from collections import deque

class FrameStack:
    """Class quản lý stack các frame liên tiếp"""
    
    def __init__(self, num_frames=4):
        """
        Khởi tạo frame stack
        
        Args:
            num_frames: số frame để stack
        """
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)
    
    def reset(self, frame):
        """Reset frame stack với frame mới"""
        for _ in range(self.num_frames):
            self.frames.append(frame)
        return self._get_state()
    
    def add(self, frame):
        """Thêm frame mới vào stack"""
        self.frames.append(frame)
        return self._get_state()
    
    def _get_state(self):
        """Lấy state từ các frame đã stack"""
        return np.array(self.frames)

class FrameProcessor:
    """Class xử lý frame"""
    
    def __init__(self, frame_size=(84, 84)):
        """
        Khởi tạo frame processor
        
        Args:
            frame_size: kích thước frame sau khi resize
        """
        self.frame_size = frame_size
    
    def process(self, frame):
        """
        Xử lý một framebatch-size
        
        Args:
            frame: array RGB
            
        Returns:
            numpy array: frame đã được xử lý
        """
        # Chuyển thành grayscale
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Resize
        frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_AREA)
        
        return frame

class PacmanEnv:
    """Môi trường wrapper cho Pacman"""
    
    def __init__(self, env_name="Pacman-v0", render_mode=None, stack_frames=4):
        """
        Khởi tạo môi trường Pacman
        
        Args:
            env_name: tên môi trường
            render_mode: chế độ render
            stack_frames: số frame để stack
        """
        # Khởi tạo môi trường gốc
        self.env = gym.make(env_name, render_mode=render_mode)
        
        # Frame processor và stacker
        self.processor = FrameProcessor(frame_size=(84, 84))
        self.stacker = FrameStack(num_frames=stack_frames)
        
        # Lưu trữ không gian hành động và quan sát
        self.action_space = self.env.action_space
        
        # Định nghĩa không gian quan sát mới sau khi stack frame
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=255,
            shape=(stack_frames, 84, 84),
            dtype=np.uint8
        )
    
    def reset(self):
        """Reset môi trường"""
        obs, info = self.env.reset()
        # Xử lý frame
        processed_obs = self.processor.process(obs)
        # Reset frame stacker với frame đầu tiên
        stacked_state = self.stacker.reset(processed_obs)
        return stacked_state, info
    
    def step(self, action):
        """Thực hiện một hành động trong môi trường"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Xử lý frame
        processed_obs = self.processor.process(obs)
        # Thêm frame mới vào stack
        stacked_state = self.stacker.add(processed_obs)
        return stacked_state, reward, terminated, truncated, info
    
    def render(self):
        """Render môi trường"""
        return self.env.render()
    
    def close(self):
        """Đóng môi trường"""
        self.env.close()