import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from basic_MDP.environment import GridMapEnvironment
from basic_MDP.environment import StateTransitEnvironment
from basic_MDP.environment import Action

class Planner():
    def __init__(self, env):
        self.env = env
        self.log = []

        self.state_list = self.env.states

        if isinstance(self.env, GridMapEnvironment)==True:
            self.env_type = 'grid_map'
        if isinstance(self.env, StateTransitEnvironment)==True:
            self.env_type = 'state_transit_diagram'

    def initialize(self):
        self.env.reset()
        self.log = []

    def plan(self, gamma=0.9, threshold=0.0001):
        raise Exception("Planner have to implements plan method.")

    def transitions_at(self, state, action):
        transition_probs = self.env.transit_func(state, action)
        for next_state in transition_probs:
            prob = transition_probs[next_state]
            reward, _ = self.env.reward_func(next_state)
            yield prob, next_state, reward

    def dict_to_grid(self, state_reward_dict):
        grid = []
        for i in range(self.env.row_length):
            row = [0] * self.env.column_length
            grid.append(row)
        for s in state_reward_dict:
            grid[s.row][s.column] = state_reward_dict[s]

        return grid


    def draw_single_grid_map_values(self, ax, grid_map_value, grid_map_policy=None, if_draw_policies=True):
        for i in range(self.env.row_length):
            for j in range(self.env.column_length):

                if (self.env.grid[i][j] == 1):
                    ax.add_patch(mpatches.Rectangle((j, self.env.row_length - i - 1), 1, 1, fc='mediumaquamarine'))
                    continue

                elif (self.env.grid[i][j] == 9):
                    ax.add_patch(mpatches.Rectangle((j, self.env.row_length - i - 1), 1, 1, fc='silver'))
                    continue

                elif (self.env.grid[i][j] == -1):
                    ax.add_patch(mpatches.Rectangle((j, self.env.row_length - i - 1), 1, 1, fc='red'))
                    continue

                center_x = 0.5 + j
                center_y = self.env.row_length - 0.5 - i

                if grid_map_policy is not None:
                    action_value_dict = grid_map_policy[self.state_list[i * self.env.row_length + j]]

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

                ax.add_patch(mpatches.Rectangle((j, self.env.row_length - i - 1), 1, 1, alpha=max(0, grid_map_value[i][j]), fc='darkorange'))
                plt.text(center_x, center_y, str(round(grid_map_value[i][j], 2)), size=10, ha='center', va='center', color='k')


        return ax




class ValueIterationPlanner(Planner):
    def __init__(self, env):
        super().__init__(env)

    def plan(self, gamma=0.9, threshold=0.001):
        self.initialize()
        actions = self.env.actions
        V = {}
        for s in self.env.states:
            # Initialize each state's expected reward.
            V[s] = 0

        while True:
            delta = 0
            self.log.append(self.dict_to_grid(V))
            for s in V:
                if not self.env.can_action_at(s):
                    continue
                expected_rewards = []
                for a in actions:
                    r = 0
                    for prob, next_state, reward in self.transitions_at(s, a):
                        r += prob * (reward + gamma * V[next_state])
                    expected_rewards.append(r)
                max_reward = max(expected_rewards)
                delta = max(delta, abs(max_reward - V[s]))
                V[s] = max_reward

            yield self.dict_to_grid(V)

            if delta < threshold:
                break
    def draw_grid_map_vlaue_iteration_planning(self):
        fig = plt.figure(figsize=(21, 6))
        fig.suptitle("Value iteration", fontsize=25)
        # plt.subplots_adjust(left=0, right=1, bottom=0.0, top=1, wspace=1, hspace=1)
        plt.subplots_adjust(wspace=0.3, hspace=0.3)

        value_list = []

        for cnt, value in enumerate(self.plan()):
            value_list.append(value)


        for cnt, value in enumerate(value_list):
            # print("Iteration: " + str(cnt + 1))
            ax = fig.add_subplot(2, len(value_list)//2, cnt + 1)
            num_font_size = 10
            ax = self.draw_single_grid_map_values(ax, value)

            # 目盛りと枠の非表示
            ax.tick_params(axis='both', which='both', bottom='off', top='off',
                           labelbottom='off', right='off', left='off', labelleft='off')

            ax.axis([0, self.env.column_length, 0, self.env.row_length])
            ax.set_xticks(np.array(range(self.env.column_length)) + 1)
            ax.set_yticks(np.array(range(self.env.row_length)) + 1)
            ax.grid(color='k', linewidth=2.0)
            ax.title.set_text('Iteration: ' + str(cnt + 1))

        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=mpl.cm.Oranges, norm=norm)
        sm.set_array([])
        plt.subplots_adjust(right=1)
        cbar_ax = fig.add_axes([0.12, -0.025, 0.7, 0.05])
        cbar = plt.colorbar(sm,
                            ticks=[0, 0.5, 1],
                            orientation='horizontal'
                            , cax=cbar_ax,
                            norm=norm)
        cbar.ax.set_xticklabels(['Low value', 'Medium value', 'High vlaue'], fontsize=20)  # horizontal colorbar
        grey_patch = mpatches.Patch(color='mediumaquamarine', label='Treasure')
        orange_patch = mpatches.Patch(color='silver', label='Block')
        purple_patch = mpatches.Patch(color='red', label='Danger')

        loc = [1.1, -2.5]
        plt.legend(handles=[grey_patch, orange_patch, purple_patch], fontsize=20, loc=loc)
        plt.savefig("value_iter_heatmaps.png", bbox_inches='tight')
        plt.show()

class PolicyIterationPlanner(Planner):
    def __init__(self, env):
        super().__init__(env)
        self.policy = {}

    def initialize(self):
        super().initialize()
        self.policy = {}
        actions = self.env.actions
        states = self.env.states
        for s in states:
            self.policy[s] = {}
            for a in actions:
                # Initialize policy.
                # At first, each action is taken uniformly.
                self.policy[s][a] = 1 / len(actions)

    def estimate_by_policy(self, gamma, threshold):
        V = {}
        for s in self.env.states:
            # Initialize each state's expected reward.
            V[s] = 0

        while True:
            delta = 0
            for s in V:
                expected_rewards = []
                for a in self.policy[s]:
                    action_prob = self.policy[s][a]
                    r = 0
                    for prob, next_state, reward in self.transitions_at(s, a):
                        r += action_prob * prob * \
                             (reward + gamma * V[next_state])
                    expected_rewards.append(r)
                value = sum(expected_rewards)
                delta = max(delta, abs(value - V[s]))
                V[s] = value
            if delta < threshold:
                break

        return V

    def plan(self, gamma=0.9, threshold=0.001):
        self.initialize()
        states = self.env.states
        actions = self.env.actions

        def take_max_action(action_value_dict):
            return max(action_value_dict, key=action_value_dict.get)

        cnt = 1
        while True:
            cnt = cnt + 1
            update_stable = True
            # Estimate expected rewards under current policy.
            V = self.estimate_by_policy(gamma, threshold)
            self.log.append(self.dict_to_grid(V))

            for s in states:
                # Get an action following to the current policy.
                policy_action = take_max_action(self.policy[s])

                # Compare with other actions.
                action_rewards = {}
                for a in actions:
                    r = 0
                    for prob, next_state, reward in self.transitions_at(s, a):
                        r += prob * (reward + gamma * V[next_state])
                    action_rewards[a] = r

                best_action = take_max_action(action_rewards)
                if policy_action != best_action:
                    update_stable = False

                # Update policy (set best_action prob=1, otherwise=0 (greedy))
                for a in self.policy[s]:
                    prob = 1 if a == best_action else 0
                    self.policy[s][a] = prob

            V_grid = self.dict_to_grid(V)
            yield V_grid, self.policy

            if update_stable:
                # If policy isn't updated, stop iteration
                break

    def draw_grid_map_policy_iteration_planning(self):
        fig = plt.figure(figsize=(24, 8))
        fig.suptitle("Policy iteration", fontsize=40)


        # pol_iter_plan = self.plan()


        value_plan_list = []

        for cnt, (value, policy) in enumerate(self.plan()):
            value_plan_list.append((value, policy))



        for cnt, (value, policy) in enumerate(value_plan_list):
            ax = fig.add_subplot(1, len(value_plan_list), cnt + 1)
            ax = self.draw_single_grid_map_values(ax, value, grid_map_policy=policy)

            plt.tick_params(axis='both', which='both', bottom='off', top='off',
                            labelbottom='off', right='off', left='off', labelleft='off')

            ax.axis([0, self.env.column_length, 0, self.env.row_length])
            ax.set_xticks(np.array(range(self.env.column_length)) + 1)
            ax.set_yticks(np.array(range(self.env.row_length)) + 1)
            ax.grid(color='k', linewidth=2.0)
            ax.title.set_text('Iteration: ' + str(cnt + 1))

        colors = ["white", "darkorange"]
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=mpl.cm.Oranges, norm=norm)
        sm.set_array([])
        plt.subplots_adjust(right=1)
        cbar_ax = fig.add_axes([0.12, -0.025, 0.7, 0.05])
        cbar = plt.colorbar(sm,
                            ticks=[0, 0.5, 1],
                            orientation='horizontal'
                            , cax=cbar_ax,
                            norm=norm)
        cbar.ax.set_xticklabels(['Low value', 'Medium value', 'High vlaue'], fontsize=20)  # horizontal colorbar
        grey_patch = mpatches.Patch(color='mediumaquamarine', label='Treasure')
        orange_patch = mpatches.Patch(color='silver', label='Block')
        purple_patch = mpatches.Patch(color='red', label='Danger')
        arrow_patch = mpatches.Arrow(0, 0, 0, 0, color='black', label='Policy')

        # arrow = matplotlib.patches.Arrow(color='black',label='My label')
        loc = [1.1, -2]
        plt.legend(handles=[arrow_patch, grey_patch, orange_patch, purple_patch], fontsize=20, loc=loc)
        plt.savefig("policy_iter_heatmaps.png", bbox_inches='tight')
        plt.show()


