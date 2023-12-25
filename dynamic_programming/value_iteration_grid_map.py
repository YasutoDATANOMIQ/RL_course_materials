import numpy as np
import matplotlib.pyplot as plt

from basic_MDP.environment import GridMapEnvironment
from DP_planner import ValueIterationPlanner


if __name__ == '__main__':
    block = np.nan
    # Setting up a grid map environment
    grid = np.array([[0, 0, 0, 0, 0, 0],
                     [0, block, 1, 0, 0, 0],
                     [0, block, block, -1, 0, 0],
                     [0, -1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0]
                     ])
    grid_map_env = GridMapEnvironment(grid)
    val_iter_palanner = ValueIterationPlanner(grid_map_env)
    val_iter_palanner.initialize()

    val_iter_palanner.draw_grid_map_vlaue_iteration_planning()
