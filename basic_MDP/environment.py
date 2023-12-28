from enum import Enum
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class State():

    def __init__(self, row=-1, column=-1):
        self.row = row
        self.column = column

    def __repr__(self):
        return "<State: [{}, {}]>".format(self.row, self.column)

    def clone(self):
        return State(self.row, self.column)

    def __hash__(self):
        return hash((self.row, self.column))

    def __eq__(self, other):
        if(type(other)==str):
            return False
        return self.row == other.row and self.column == other.column


class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

opposite_direction_dict = {0:1, 1:0, 2:3, 3:2}


# TODO: implementing a MDP environment with state transition model
class StateTransitEnvironment():
    pass



class GridMapEnvironment():

    def __init__(self, grid, move_prob=0.8):
        # grid is 2d-array. Its values are treated as an attribute.
        # Kinds of attribute is following.
        #  0: ordinary cell
        #  -1: damage cell (game end)
        #  1: reward cell (game end)
        #  np.nan: block cell (can't locate agent)
        self.grid = grid
        self.agent_state = State()

        self.state_idx_dict = {s:i for i, s in enumerate(self.states)}

        # Default reward is minus. Just like a poison swamp.
        # It means the agent has to reach the goal fast!
        self.default_reward = -0.04

        # Agent can move to a selected direction in move_prob.
        # It means the agent will move different direction
        # in (1 - move_prob).
        self.move_prob = move_prob
        self.reset()

    @property
    def row_length(self):
        return len(self.grid)

    @property
    def column_length(self):
        return len(self.grid[0])

    @property
    def actions(self):
        # 0: up
        # 1: down
        # 2: left
        # 3: right
        return list(range(4))

    @property
    def states(self):
        states = []
        for row in range(self.row_length):
            for column in range(self.column_length):
                # Block cells are not included to the state.
                if np.isnan(self.grid[row][column])==False:
                    states.append(State(row, column))
        return states

    def transit_func(self, state, action):
        transition_probs = {}
        if not self.can_action_at(state):
            # Already on the terminal cell.
            return transition_probs

        opposite_direction = opposite_direction_dict[action]

        for a in self.actions:
            prob = 0
            if a == action:
                prob = self.move_prob
            elif a != opposite_direction:
                prob = (1 - self.move_prob) / 2

            next_state = self._move(state, a)
            if next_state not in transition_probs:
                transition_probs[next_state] = prob
            else:
                transition_probs[next_state] += prob

        return transition_probs

    def can_action_at(self, state):
        if self.grid[state.row][state.column] == 0:
            return True
        else:
            return False

    def _move(self, state, action):
        if not self.can_action_at(state):
            raise Exception("Can't move from here!")

        next_state = state.clone()

        # Execute an action (move).
        if action==0:
            next_state.row -= 1
        elif action==1:
            next_state.row += 1
        elif action==2:
            next_state.column -= 1
        elif action==3:
            next_state.column += 1

        # Check whether a state is out of the grid.
        if not (0 <= next_state.row < self.row_length):
            next_state = state
        if not (0 <= next_state.column < self.column_length):
            next_state = state

        # Check whether the agent bumped a block cell.
        if np.isnan(self.grid[next_state.row][next_state.column])==True:
            next_state = state

        return next_state

    def reward_func(self, state):
        reward = self.default_reward
        done = False

        # Check an attribute of next state.
        attribute = self.grid[state.row][state.column]
        if attribute == 1:
            # Get reward! and the game ends.
            reward = 1
            done = True
        elif attribute == -1:
            # Get damage! and the game ends.
            reward = -1
            done = True

        return reward, done

    def reset(self):
        # Locate the agent at lower left corner.
        self.agent_state = State(self.row_length - 1, 0)


        return self.state_idx_dict[self.agent_state]

    def step(self, action):
        next_state, reward, done = self.transit(self.agent_state, action)
        if next_state is not None:
            self.agent_state = next_state
        next_state_idx = self.state_idx_dict[next_state]
        return next_state_idx, reward, done, None, None

    def transit(self, state, action):
        transition_probs = self.transit_func(state, action)
        if len(transition_probs) == 0:
            return None, None, True

        next_states = []
        probs = []
        for s in transition_probs:
            next_states.append(s)
            probs.append(transition_probs[s])

        next_state = np.random.choice(next_states, p=probs)
        reward, done = self.reward_func(next_state)
        return next_state, reward, done


    def draw_single_grid_map_values(self, ax, cnt=None,
                                    grid_map_value=None,
                                    grid_map_policy=None,
                                    grid_action_value=None,
                                    grid_agent_state_idx=None
                                    ):




        for i in range(self.row_length):
            for j in range(self.column_length):


                # temp_action_value = grid_action_value[i, j]

                if (self.grid[i][j]==1):
                    ax.add_patch(mpatches.Rectangle((j, self.row_length - i - 1), 1, 1, fc='mediumaquamarine'))
                    continue

                elif np.isnan(self.grid[i][j]):
                    ax.add_patch(mpatches.Rectangle((j, self.row_length - i - 1), 1, 1, fc='silver'))
                    continue

                elif (self.grid[i][j]==-1):
                    ax.add_patch(mpatches.Rectangle((j, self.row_length - i - 1), 1, 1, fc='red'))
                    continue

                center_x = 0.5 + j
                center_y = self.row_length - 0.5 - i


                # print("grid_action_value")
                # print(grid_action_value)
                if grid_action_value is not None:

                    action_value_fontsize=7.5

                    temp_state = State(i, j)




                    temp_action_value = grid_action_value[self.state_idx_dict[temp_state]]
                    # print('temp_action_value: {}'.format(temp_action_value))
                    ax.add_patch(mpatches.Rectangle((j, self.row_length - i - 1), 1, 1, fc='white'))
                    # ax.add_patch(
                    #     mpatches.Rectangle((j, self.row_lenght - state.row - 1), 1, 1, alpha=max(0, value_mean),
                    #               fc='darkorange'))
                    ax.text(center_x, center_y + 0.2, str(np.round(temp_action_value[0], 4)), fontsize=action_value_fontsize)
                    ax.text(center_x, center_y - 0.2, str(np.round(temp_action_value[1], 4)), fontsize=action_value_fontsize)
                    ax.text(center_x + 0.2, center_y, str(np.round(temp_action_value[2], 4)), fontsize=action_value_fontsize)
                    ax.text(center_x - 0.2, center_y, str(np.round(temp_action_value[3], 4)), fontsize=action_value_fontsize)
                    continue

                if grid_map_value is not None:
                    ax.add_patch(mpatches.Rectangle((j, self.row_length - i - 1), 1, 1, alpha=max(0, grid_map_value[i][j]), fc='darkorange'))
                    plt.text(center_x, center_y, str(round(grid_map_value[i][j], 2)), size=10, ha='center', va='center', color='k')


                    if grid_map_policy is not None:
                        action_value_dict = grid_map_policy[self.states[i * self.row_length + j]]

                        up = action_value_dict[0]
                        down = action_value_dict[1]
                        left = action_value_dict[2]
                        right = action_value_dict[3]

                        plt.arrow(center_x, center_y + 0.2, 0.0, 0.15 * up, width=0.025 * up, head_width=0.075 * up,
                                  head_length=0.1 * up, fc='k', ec='k')
                        plt.arrow(center_x, center_y - 0.2, 0.0, -0.15 * down, width=0.025 * down, head_width=0.075 * down,
                                  head_length=0.1 * down, fc='k', ec='k')
                        plt.arrow(center_x + 0.2, center_y, 0.15 * right, 0.0, width=0.025 * right, head_width=0.075 * right,
                                  head_length=0.1 * right, fc='k', ec='k')
                        plt.arrow(center_x - 0.2, center_y, -0.15 * left, 0.0, width=0.025 * left, head_width=0.075 * left,
                                  head_length=0.1 * left, fc='k', ec='k', label='Lokale Orientierung')

                # if grid_action_value is not None:

        # Temporal naive error handling
        try:
            grid_agent_state = self.states[grid_agent_state_idx]
        except:
            grid_agent_state = None


        if grid_agent_state is not None:
        #     agent_state_idx = self.state_idx_dict[grid_agent_state]
            agent_center_x = 0.45 + grid_agent_state.column
            agent_center_y = self.row_length - 0.5 - grid_agent_state.row
            ax.add_patch(mpatches.Circle((agent_center_x, agent_center_y), 0.25, fc='grey'))
            # ax.add_patch(mpatches.Circle((agent_center_x, self.row_length - agent_center_y - 1), 1, fc='grey'))



        # 目盛りと枠の非表示
        ax.tick_params(axis='both', which='both', bottom='off', top='off',
                               labelbottom='off', right='off', left='off', labelleft='off')

        ax.axis([0, self.column_length, 0, self.row_length])
        ax.set_xticks(np.array(range(self.column_length)) + 1)
        ax.set_yticks(np.array(range(self.row_length)) + 1)
        ax.grid(color='k', linewidth=2.0)
        # ax.title.set_text('Iteration: ' + str(cnt + 1))

        return ax