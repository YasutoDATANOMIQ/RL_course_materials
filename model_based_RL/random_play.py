from env_tic_tac_toe import State
from env_tic_tac_toe import random_action
from tree_search import mini_max_action
# ミニマックス法とランダムで対戦

# 状態の生成
state = State()

# ゲーム終了までのループ
while True:
    # ゲーム終了時
    if state.is_done():
        break

    # 行動の取得
    if state.is_first_player():
        action = mini_max_action(state)
    else:
        action = random_action(state)

    # 次の状態の取得
    state = state.next(action)

    # 文字列表示
    print(state)
    print()