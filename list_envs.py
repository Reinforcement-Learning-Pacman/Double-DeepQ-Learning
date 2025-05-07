import gymnasium as gym

# Check if env exist in ale_py
all_envs = gym.envs.registry.keys()

atari_envs = [env_id for env_id in all_envs if env_id.startswith('ale_py:ALE/')]
print("All Atari environments:")
for env_id in atari_envs:
    print(f" - {env_id}")
