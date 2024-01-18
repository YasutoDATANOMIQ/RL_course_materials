import numpy as np
import matplotlib.pyplot as plt
import gym

from basic_MDP.environment import GridMapEnvironment
from model_free_RL.tabular_model_free_agent import QLearningAgent
from model_free_RL.tabular_model_free_agent import SARSAAgent
from model_free_RL.tabular_model_free_agent import repetive_experiment
from model_free_RL.tabular_model_free_agent import compare_experiments

if __name__ == '__main__':
    goal = 1
    fall = -100
    # Setting a cliff walking environment
    grid = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, fall, fall, fall, fall, fall, fall, fall, fall, fall, fall,goal]
                     ])

    # Setting up an enviornment
    grid_map_env = GridMapEnvironment(grid, move_prob=1, constant_reward=-1)

    # Setting up agents to compare
    ql_agent = QLearningAgent(grid_map_env, epsilon=0.1)
    sarsa_agent = SARSAAgent(grid_map_env, epsilon=0.1)
    el_agent_list = [ql_agent, sarsa_agent]
    legend_label_list = ['Q-Learning', 'SARSA']
    plot_color_list = ['red', 'blue']


    # Conducting experiments with different agents.
    episode_count = 500
    experiment_num = 100
    compare_experiments(el_agent_list, legend_label_list, experiment_num, episode_count, grid_map_env)

