import gymnasium as gym

# Check if env exist in ale_py
all_envs = gym.envs.registry.keys()

print("All environments:")
for env_id in all_envs:
    print(f" - {env_id}")
