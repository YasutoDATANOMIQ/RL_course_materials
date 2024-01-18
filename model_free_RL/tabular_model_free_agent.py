import itertools
import random
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from PIL import Image
import io


def repetive_experiment(el_agent, experiment_num, episode_count, grid_map_env):
    reward_log_list = []
    for i in range(experiment_num):
        el_agent.learn(grid_map_env, episode_count=episode_count, render=False,)
        reward_log_list.append(el_agent.reward_log)

    # Calculate mean and standard deviation for each position across arrays
    reward_average = np.mean(reward_log_list, axis=0)
    reward_std = np.std(reward_log_list, axis=0)

    return reward_average, reward_std

def compare_experiments(el_agent_list, legend_label_list, experiment_num, episode_count, grid_map_env):

    # Getting random colors
    # TODO: I want the colors to be more distinguishable
    colors = plt.get_cmap("tab20").colors
    colors = list(colors)
    random.shuffle(colors)



    plt.figure()
    for idx, (el_agent, legend_label) in enumerate(zip(el_agent_list, legend_label_list)):
        print("Conducting experiments of {} for {} episodes, {} times".format(legend_label, episode_count, experiment_num))
        reward_average, reward_std = repetive_experiment(el_agent, experiment_num, episode_count, grid_map_env)

        plot_color = colors[idx]

        # Plot average
        plt.plot(reward_average, label=legend_label, color=plot_color)

        # Use fill_between to visualize standard deviation
        plt.fill_between(range(len(reward_average)),
                         reward_average - reward_std,
                         reward_average + reward_std,
                         alpha=0.1,
                         color=plot_color,
                         )

    plt.legend()
    plt.show()


class ELAgent():
    # TODO: also to generalize action taking processes

    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.history = []

    def initialize_Q_value(self, state_space_size, action_space_size, max_reward=None):
        """
        Initializes the initial action values. When a max reward is give, the action values are optimistically
        initialized with the maz reward, otherwise just with 0 values.
        :param state_space_size:
        :param action_space_size:
        :param max_reward:
        :return:
        """

        if max_reward is None:
            initialized_Q = np.zeros((state_space_size, action_space_size))
        else:
            initialized_Q = np.ones((state_space_size, action_space_size))*max_reward

        return initialized_Q

    def policy(self, s, actions, utility_function='epsilon_greedy'):
        """
        Returns an action given a state.
        TODO: To integrate softmax policy with temperature, action selection with UCB.
        :param s:
        :param actions:
        :return:
        """

        if type(s)==tuple:
            s = s[0]

        if np.random.random() < self.epsilon:
            return np.random.randint(len(actions))
        else:
            if s < len(self.Q) and sum(self.Q[s])!=0:
                return np.argmax(self.Q[s])
            else:
                return np.random.randint(len(actions))

    def init_log(self):
        self.reward_log = []

    def log(self, reward):
        self.reward_log.append(reward)

    def show_reward_log(self, interval=50):
        """
        Plots a logs of average reward gained with standard deviations
        :param interval:
        :param episode:
        :return:
        """

        # TODO: the part below should not be in this function
        # if episode > 0:
        #     rewards = self.reward_log[-interval:]
        #     mean = np.round(np.mean(rewards), 3)
        #     std = np.round(np.std(rewards), 3)
        #     print("At Episode {} average reward is {} (+/-{}).".format(
        #            episode, mean, std))

        if True:
            indices = list(range(0, len(self.reward_log), interval))
            means = []
            stds = []
            for i in indices:
                rewards = self.reward_log[i:(i + interval)]
                means.append(np.mean(rewards))
                stds.append(np.std(rewards))
            means = np.array(means)
            stds = np.array(stds)
            plt.figure()
            plt.title("Reward History")
            plt.grid()
            plt.fill_between(indices, means - stds, means + stds, alpha=0.05, color="g")
            plt.plot(indices, means, "o-", color="g", label="Rewards for each {} episode".format(interval))
            plt.legend(loc="best")
            plt.show()



    def extract_episodes_to_animate(self, first_episode_idx, last_episode_idx, episode_interval):
        """
        # TODO: It will be more efficient to return only indexes of episodes rather thatn history of episode itself.
        :param first_episode_idx:
        :param last_episode_idx:
        :param episode_interval:
        :return:
        """
        episode_range = range(first_episode_idx, last_episode_idx, episode_interval)
        episode_slice = slice(first_episode_idx, last_episode_idx, episode_interval)

        episodes_to_animate = self.history[episode_slice]
        episode_number_list = [[episode_number] * len(episodes_to_animate[cnt]) for (cnt, episode_number) in enumerate(episode_range)]
        episode_number_flattened = list(itertools.chain.from_iterable(episode_number_list))
        episode_index_list = [list(range(len(episodes_to_animate[cnt]))) for (cnt, episode_number) in enumerate(episode_range)]
        episode_index_flattened = list(itertools.chain.from_iterable(episode_index_list))
        episodes_to_animate_flattened = list(itertools.chain.from_iterable(episodes_to_animate))

        return episodes_to_animate_flattened, episode_number_flattened, episode_index_flattened

    def draw_grid_map_trial_and_errors(self, first_episode_idx, last_episode_idx, episode_interval):
        """
        Exports a gif animation of extracted frames.
        Just draws each frame as a matplotlib figure.
        :param first_episode_idx:
        :param last_episode_idx:
        :param episode_interval:
        :return:
        """
        extracted_history_flattened,\
            episode_number_flattened, \
            episode_index_flattened= self.extract_episodes_to_animate(first_episode_idx, last_episode_idx, episode_interval)

        value_list = []

        width, height = 200, 200

        frames = []

        for frame_idx, (agent_state, action_value) in enumerate(extracted_history_flattened):
            plt.clf()
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(1, 1, 1)

            num_font_size = 10
            ax = self.env.draw_single_grid_map_values(ax, grid_action_value=action_value, grid_agent_state_idx=agent_state)

            episode_No = episode_number_flattened[frame_idx]
            step_idx = episode_index_flattened[frame_idx]

            plt.title("Episode: {}, step: {}".format(episode_No, step_idx))

            # Save the Matplotlib plot to a BytesIO object
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)

            # Use Pillow to open the image from the BytesIO object
            pillow_image = Image.open(buffer)

            frames.append(pillow_image)


        # TODO: This might not be the best way to make a gif animation. All the matplotlib windows are open.
        frames[0].save('animated.gif', format='GIF', append_images=frames[1:], save_all=True, duration=500, loop=0)




class QLearningAgent(ELAgent):
    def __init__(self, env, epsilon=0.1):
        super().__init__(epsilon)
        self.env = env

    #TODO: To dynamically control the learning rate
    def learn(self, env, episode_count=1000, gamma=0.9, learning_rate=0.1, render=False, report_interval=50, max_reward=None):
        self.init_log()

        try: # First trying to get actions in OpenAI Gym format
            actions = list(range(env.action_space.n))
        except: # Otherwise getting actions in the custom grid map environment formats
            actions =  list(range(len(env.actions)))

        try: # First trying to get actions in OpenAI Gym format
            self.Q = self.initialize_Q_value(env.observation_space.n, env.action_space.n, max_reward=max_reward)
        except: # Otherwise getting actions in the custom grid map environment formats
            self.Q = self.initialize_Q_value(len(self.env.states), len(self.env.actions), max_reward=max_reward)




        for episode_idx in range(episode_count):
            history_one_episode = []

            # env.reset() sometimes returns a state as a tuple
            # TODO: the case handling belwo might not be the best
            s = env.reset()
            if type(s) == tuple:
                s = s[0]

            # Appending a history of Q-value
            # TODO: This migth slower the process, so this should be turned of if animationis not needed
            history_one_episode.append((s, np.copy(self.Q)))

            done = False
            while not done:

                # Drawing a step
                if render:
                    env.render()

                # Taking an action
                a = self.policy(s, actions)
                # Getting the next state, reward
                n_state, reward, done, info, _ = env.step(a)

                # Calculating a TD error
                gain = reward + gamma * max(self.Q[n_state])
                estimated = self.Q[s][a]
                TD_loss = gain - estimated

                # Action value update
                self.Q[s][a] += learning_rate * TD_loss

                # Recording a history
                history_one_episode.append((n_state, np.copy(self.Q)))

                # Shifting to the next state
                s = n_state

            else:
                self.log(reward)

            self.history.append(history_one_episode)


class DoubleQLearningAgent(ELAgent):
    def __init__(self, env, epsilon=0.1):
        super().__init__(epsilon)
        self.env = env



    #TODO: To dynamically control the learning rate
    def learn(self, env, episode_count=1000, gamma=0.9, learning_rate=0.1, render=False, report_interval=50,
              max_reward=None):
        self.init_log()
        # max_reward = 100
        try: # First trying to get actions in OpenAI Gym format
            actions = list(range(env.action_space.n))
        except: # Otherwise getting actions in the custom grid map environment formats
            actions =  list(range(len(env.actions)))

        try: # First trying to get actions in OpenAI Gym format
            self.Q_1 = self.initialize_Q_value(env.observation_space.n, env.action_space.n, max_reward=max_reward)
            self.Q_2 = self.initialize_Q_value(env.observation_space.n, env.action_space.n, max_reward=max_reward)
            self.Q = self.Q_1 + self.Q_2
        except: # Otherwise getting actions in the custom grid map environment formats
            self.Q_1 = self.initialize_Q_value(len(self.env.states), len(self.env.actions), max_reward=max_reward)
            self.Q_2 = self.initialize_Q_value(len(self.env.states), len(self.env.actions), max_reward=max_reward)
            self.Q = self.Q_1 + self.Q_2

        for episode_idx in range(episode_count):
            history_one_episode = []

            # env.reset() sometimes returns a state as a tuple
            # TODO: the case handling belwo might not be the best
            s = env.reset()
            if type(s) == tuple:
                s = s[0]

            # Appending a history of Q-value
            # TODO: This migth slower the process, so this should be turned of if animationis not needed
            history_one_episode.append((s, np.copy(self.Q)))

            done = False
            while not done:

                # Drawing a step
                if render:
                    env.render()

                # Taking an action
                a = self.policy(s, actions)
                # Getting the next state, reward
                n_state, reward, done, info, _ = env.step(a)

                # In Double Q-learning,
                if np.random.random() < 0.5:
                    # Calculating a TD error
                    action_to_update = np.argmax(self.Q_1[n_state])
                    gain = reward + gamma * self.Q_2[n_state][action_to_update]
                    estimated = self.Q_1[s][a]
                    TD_loss = gain - estimated
                    # Action value update
                    self.Q_1[s][a] += learning_rate * TD_loss
                else:
                    # Calculating a TD error
                    action_to_update = np.argmax(self.Q_2[n_state])
                    gain = reward + gamma * self.Q_1[n_state][action_to_update]
                    estimated = self.Q_2[s][a]
                    TD_loss = gain - estimated
                    # Action value update
                    self.Q_2[s][a] += learning_rate * TD_loss

                self.Q = self.Q_1 + self.Q_2
                # Recording a history
                history_one_episode.append((n_state, np.copy(self.Q)))

                # Shifting to the next state
                s = n_state

            else:
                self.log(reward)

            self.history.append(history_one_episode)
