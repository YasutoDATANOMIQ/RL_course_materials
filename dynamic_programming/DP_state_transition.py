import copy
import numpy as np

env_dict = {'Flat': {
        # From flat to flat: It's easy. I just had to sleep, but I got no reward. I cannot work home.
        'To Flat': {1: {'resulting_state': 'Flat', 'reward': 0}},
        # I had an obesseion to go to Starbucks, so with a 30% probability, I ended up going to Starbucks.
        'To Lab': {0.7: {'resulting_state': 'Lab', 'reward': 10}, 0.3: {'resulting_state': 'Starbucks', 'reward': 2}},
        # I chose a room clsoe to Starbucks, so going there was quite easy.
        'To Starbucks': {1: {'resulting_state': 'Starbucks', 'reward': 2}}
    },

        'Lab': {
            # It was quite easy to go home, but I won't get any rewards. I did not have a desk in my tiny flat.
            'To Flat': {1: {'resulting_state': 'Flat', 'reward': 0}},
            # Simply working in lab gets tedious and tiresome. I might keep working there after feeling refreshed by taking a walk.
            # But it often fails and I go back on my way home start working again in Starbucks.
            # Sometimes I'm simply tired and end up going home.
            'To Lab': {0.6: {'resulting_state': 'Lab', 'reward': 15},
                       0.3: {'resulting_state': 'Starbucks', 'reward': 3},
                       0.1: {'resulting_state': 'Flat', 'reward': 0}},
            # Going to Starbucks again is relatively easy and with receipt of coffee, next coffee was cheaper.
            'To Starbucks': {0.9: {'resulting_state': 'Starbucks', 'reward': 4},
                             0.1: {'resulting_state': 'Flat', 'reward': 0}}
        },

        'Starbucks': {
            # The flat was really close to Starbucks, so going home was quite easy, but there is no reward.
            'To Flat': {1: {'resulting_state': 'Flat', 'reward': 0}},
            # Somehow I did not feel like going home from Starbucks to do something like taking a nap.
            # And changing a place to my lab after coffee was a actually a good habit to remain productivity.
            'To Lab': {1: {'resulting_state': 'Lab', 'reward': 12}},
            # It is not effective to tray to stay in Starbucks because it gets crowded and I need to order extra drinks.
            # Also productivity there just decreases becasuse I had no extra monitors
            'To Starbucks': {1: {'resulting_state': 'Starbucks', 'reward': -2}}
        }
    }


def get_key_of_largest_value(dict_var):
    max_value = -np.inf
    max_key = None
    for key, value in dict_var.items():
        if value > max_value:
            max_value = value
            max_key = key

    return max_value, max_key

def bellman_optimal_operation(action_dict, V_backup, gamma=0.9):
    weighted_TD_sum_dict = {}
    for action, action_prob_dict in action_dict.items():
        weighted_TD_sum = 0
        for transit_prob, result_dict in action_prob_dict.items():
            resulting_state = result_dict['resulting_state']
            reward = result_dict['reward']
            weighted_TD_sum += transit_prob * (reward + gamma * V_backup[resulting_state])
        weighted_TD_sum_dict[action] = weighted_TD_sum
    max_weighted_TD, max_action = get_key_of_largest_value(weighted_TD_sum_dict)
    return max_weighted_TD, max_action


def bellman_expectation_operation(policy_given_s, action_dict, V_backup, gamma=0.9):
    # state_value = 0
    V_s = 0
    for action, action_prob_dict in action_dict.items():
        weighted_TD_sum = 0
        for transit_prob, result_dict in action_prob_dict.items():
            resulting_state = result_dict['resulting_state']
            reward = result_dict['reward']
            weighted_TD_sum += transit_prob * (reward + gamma * V_backup[resulting_state])
        V_s += policy_given_s[action] * weighted_TD_sum

    return V_s


def policy_evaluation(policy, thresh=0.001):
    V_updated = {'Flat': 0, 'Lab': 0, 'Starbucks': 0}

    while True:
        V_backup = copy.deepcopy(V_updated)
        for state, action_dict in env_dict.items():
            policy_given_s = policy[state]
            V_updated[state] = bellman_expectation_operation(policy_given_s, action_dict, V_backup)

        V_difference_list = [abs(V_backup[s] - V_updated[s]) for s in V_backup.keys()]
        V_backup = V_updated

        print("V_backup: {}".format(V_backup))
        if max(V_difference_list) < thresh:
            return V_updated


def greedy_DP_policy_selection(env_dict, V_backup):
    updated_policy = {}
    for state, action_dict in env_dict.items():
        max_weighted_TD, max_action = bellman_optimal_operation(action_dict, V_backup)
        updated_policy[state] = {'To Flat': 0, 'To Lab': 0, 'To Starbucks': 0}
        updated_policy[state][max_action] = 1

    return updated_policy

if __name__ == '__main__':

    V_updated = {'Flat': 0, 'Lab': 1000, 'Starbucks': 0}
    thresh = 0.001

    # Value iteration
    for _ in range(100):
        # V_updated = {'Flat': 0, 'Lab': 0, 'Starbucks': 0}
        V_backup = copy.deepcopy(V_updated)

        for state, action_dict in env_dict.items():
            max_weighted_TD, max_action = bellman_optimal_operation(action_dict, V_backup)
            V_updated[state] = max_weighted_TD

        V_difference_list = [abs(V_backup[s] - V_updated[s]) for s in V_backup.keys()]

        if max(V_difference_list) < thresh:
            optimal_policy = greedy_DP_policy_selection(env_dict, V_updated)
            break

    policy = {'Flat': {'To Flat': 0.3, 'To Lab': 0.3, 'To Starbucks': 0.4},
              'Lab': {'To Flat': 0.3, 'To Lab': 0.3, 'To Starbucks': 0.4},
              'Starbucks': {'To Flat': 0.3, 'To Lab': 0.3, 'To Starbucks': 0.4}
              }

    # Policy iteration
    for _ in range(5):
        V_pi = policy_evaluation(policy)
        updated_policy = greedy_DP_policy_selection(env_dict, V_pi)
        policy = updated_policy
