from tree_search import mcts_action
from env_tic_tac_toe import random_action
from env_tic_tac_toe import play

# 任意のアルゴリズムの評価
def evaluate_algorithm_of(label, next_actions, EP_GAME_COUNT=1):
    # 複数回の対戦を繰り返す
    total_point = 0
    for i in range(EP_GAME_COUNT):
        # 1ゲームの実行
        if i % 2 == 0:
            total_point += play(next_actions)
        else:
            total_point += 1 - play(list(reversed(next_actions)))

        # 出力
        print('\rEvaluate {}/{}'.format(i + 1, EP_GAME_COUNT), end='')
    print('')

    # 平均ポイントの計算
    average_point = total_point / EP_GAME_COUNT
    print(label.format(average_point))


if __name__ == '__main__':
    # VSランダム
    next_actions = (mcts_action, random_action)
    evaluate_algorithm_of('VS_Random {:.3f}', next_actions)

    # # VSアルファベータ法
    # next_actions = (mcts_action, alpha_beta_action)
    # evaluate_algorithm_of('VS_AlphaBeta {:.3f}', next_actions)