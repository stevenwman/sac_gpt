import mani_skill.envs
import gymnasium as gym
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
import numpy as np
import torch
N = 4
env = gym.make("PickCube-v1", num_envs=N, max_episode_steps=50)
env = ManiSkillVectorEnv(env, auto_reset=True, ignore_terminations=False)
env.action_space # shape (N, D)
env.single_action_space # shape (D, )
env.observation_space # shape (N, ...)
env.single_observation_space # shape (...)
env.reset()

terminated = truncated = torch.tensor([False], dtype=torch.bool)
ep_ret = 0.
ep_len = 0.

while not (terminated | truncated).all():
    obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
    ep_ret += rew
    ep_len += ~ (terminated | truncated)
print({"test/return": torch.mean(ep_ret.cpu()), 
       "test/length": torch.mean(ep_len.cpu())})

print('hi')
# obs (N, ...), rew (N, ), terminated (N, ), truncated (N, )

# import mani_skill.envs
# import gymnasium as gym
# from mani_skill.utils.wrappers.record import RecordEpisode
# from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
# N = 4
# env = gym.make("PickCube-v1", num_envs=N, render_mode="rgb_array")
# env = RecordEpisode(env, output_dir="videos", save_trajectory=True, trajectory_name="trajectory", max_steps_per_video=50, video_fps=30)
# env = ManiSkillVectorEnv(env, auto_reset=True) # adds auto reset
# env.reset()
# for _ in range(200):
#     obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    