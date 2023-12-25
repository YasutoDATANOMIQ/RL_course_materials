import numpy as np
import matplotlib.pyplot as plt
from basic_MDP.environment import GridMapEnvironment
from DP_planner import PolicyIterationPlanner

if __name__ == '__main__':

    # Setting up a grid map environment
    grid = np.array([[0, 0, 0, 0, 0, 0],
                     [0, 9, 1, 0, 0, 0],
                     [0, 9, 9, -1, 0, 0],
                     [0, -1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0]
                     ])
    grid_map_env = GridMapEnvironment(grid)

    # Conducting policy iteration
    pol_iter_palanner = PolicyIterationPlanner(grid_map_env)
    pol_iter_palanner.initialize()

    pol_iter_palanner.draw_grid_map_policy_iteration_planning()


    # # Visualizing processes of the planning
    # draw_grid_map_policy_iteration_planning(pol_iter_palanner)