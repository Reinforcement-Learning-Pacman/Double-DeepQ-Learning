import gymnasium as gym
import ale_py

# List all registered environments
all_envs = gym.envs.registry.keys()
# Filter and display only Atari environments
atari_envs = [env_id for env_id in all_envs if env_id.startswith('ale_py:ALE/')]
print("All Atari environments:")
for env_id in atari_envs:
    print(f" - {env_id}")