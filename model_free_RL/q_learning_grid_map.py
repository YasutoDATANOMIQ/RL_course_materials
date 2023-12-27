import numpy as np
import gym


from basic_MDP.environment import GridMapEnvironment
from model_free_RL.tabular_model_free_agent import QLearningAgent
from model_free_RL.frozen_lake_util import show_q_value

if __name__ == '__main__':
    block = np.nan
    grid = np.array([[0, 0, 0, 0, 0, 0],
                     [0, block, 1, 0, 0, 0],
                     [0, block, block, -1, 0, 0],
                     [0, -1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0]
                     ])

    grid_map_env = GridMapEnvironment(grid)
    episode_count = 10
    agent = QLearningAgent(grid_map_env)
    agent.learn(grid_map_env, episode_count=episode_count, render=False, report_interval=1)
    print("len(agent.history)")
    print(len(agent.history))


    # Drawing an animation of model-free RL
    first_episode_idx = 0
    last_episode_idx = 10
    episode_interval = 1

    agent.draw_grid_map_trial_and_errors(first_episode_idx, last_episode_idx, episode_interval)



    # env = gym.make("FrozenLake-v1")
    # agent = QLearningAgent(env)
    # agent.learn(env, episode_count=500)
    # show_q_value(agent.Q)
    # agent.show_reward_log()
