# Source:
# https://github.com/triwave33/reinforcement_learning/blob/master/qiita/RL_11_linear_approx.py
# https://qiita.com/triwave33/items/78780ec37babf154137d

### -*-coding:utf-8-*-
import random
import numpy as np
import scipy.linalg as LA
from itertools import product
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import tqdm
import gym
import math
import seaborn as sns
import os
import datetime
import shutil
import sys
from collections import deque
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.optimizers import Adam, SGD, RMSprop
from keras import backend as K
import tensorflow as tf
import time

# 学習用パラメータ
GAMMA = 0.99
EPS_LAST = 0
render = 0  # 描画モード
num_episode = 301

# Mountaincar
env = gym.make('MountainCar-v0')
NUM_STATE = 2
NUM_ACTION = 3


class BasicAgent:
    def __init__(self, eps_ini, eps_last):
        self.eps_ini = eps_ini
        self.eps_last = eps_last

    def getQ(self, s, a):
        raise NotImplementedError("You have to implement getQ method.")

    def select_action(self, s, eps):
        raise NotImplementedError("You have to implement select_ation method.")

    def update(self, s, a, reward, s_dash, a_dash):
        raise NotImplementedError("You have to implement update method.")


# テーブル法用のエージェントクラス
class TableAgent(BasicAgent):
    def __init__(self, N, alpha_ini, alpha_last, eps_ini, eps_last):
        super().__init__(eps_ini, eps_last)
        self.N = N  # テーブルの分割数
        self.done = False
        # 価値関数の初期化
        self.Q = np.zeros(((NUM_ACTION,) + (self.N,) * NUM_STATE))
        self.reward_array = []
        self.min_list = np.array([-1.2, -0.07])  # グリッド分割下限値
        self.max_list = np.array([0.6, 0.07])  # グリッド分割上限値
        self.alpha_ini = alpha_ini
        self.alpha_last = alpha_last

    def digitize(self, obs):  # envからの観測値をを離散化
        if type(obs)==tuple:
            obs = obs[0]
        s = [int(np.digitize(obs[i], np.linspace(self.min_list[i], \
                                                 self.max_list[i], self.N - 1))) for i in range(NUM_STATE)]
        return s

    def getQ(self, s, a):  # s, aにおける行動価値関数を出力
        return self.Q[a, s[0], s[1]]

    def select_action(self, s, eps):
        s = self.digitize(s)
        # e-greedyによる行動選択
        if np.random.rand() < eps:  # random
            action = np.random.randint(NUM_ACTION)
            return action
        else:
            action = np.argmax(self.Q[:, s[0], s[1]])  # greedy
            # 最大値をとるアクションが複数ある場合、その中からランダムに選択
            is_greedy_index = np.where(self.Q[:, s[0], s[1]] == action)[0]
            if len(is_greedy_index) > 1:
                action = np.random.choice(is_greedy_index)
            return action

    # 価値関数の更新
    def update(self, s, a, reward, s_dash, a_dash):
        s = self.digitize(s)
        s_dash = self.digitize(s_dash)
        Qval = self.getQ(s, a)
        Qval_dash = self.getQ(s_dash, a_dash)
        self.Q[a, s[0], s[1]] = Qval + ALPHA * (reward + GAMMA * Qval_dash - Qval)


class LinearFuncAgent(BasicAgent):
    # 基底関数
    min_list = env.observation_space.low  # 下限値 [-1.2 -0.07]
    max_list = env.observation_space.high  # 上限値 [-0.6, 0.07]
    norm_factor = np.array([1.2, 0.07])  # 状態間のスケールを調整するをファクター
    norm_factor = norm_factor.reshape(len(norm_factor), 1)

    def __init__(self, N, sigma, alpha_ini, alpha_last, eps_ini, eps_last):
        super().__init__(eps_ini, eps_last)
        self.done = False
        self.s1_space = np.linspace(self.min_list[0], self.max_list[0], N)
        self.s2_space = np.linspace(self.min_list[1], self.max_list[1], N)
        self.alpha_ini = alpha_ini
        self.alpha_last = alpha_last

        b = (N ** NUM_STATE)  # 状態空間を分割した場合の総数

        # 基底関数の定数項を初期化（学習対象外）
        # 基底関数をガウス関数とし、中心値µを初期化
        self.mu_array = np.random.rand(NUM_STATE, b)  # ランダムの場合
        # self.mu_array = np.zeros((NUM_STATE, b)) # オール0の場合
        cnt = 0
        for i in self.s1_space:
            for j in self.s2_space:
                self.mu_array[0, cnt] = i
                self.mu_array[1, cnt] = j
                cnt += 1
        # 分散を初期化（固定かつ共通）
        self.sigma = 0.1

        # 3つのアクションに対して、同じ基底関数セットを用いる
        self.mu_list = [np.copy(self.mu_array)] * NUM_ACTION

        # 学習対象のパラメータの初期化
        self.theta_list = [np.zeros(b), np.zeros(b), np.zeros(b)]

    # 基底関数(入力:状態空間(2次元)、出力:基底関数の出力(b*3次元))
    def rbfs(self, s):
        if type(s)==tuple:
            s = s[0]
        s = s.reshape(len(s), 1)  # 2次元に整形
        return np.exp(-np.square(LA.norm((self.mu_list - s) / self.norm_factor, axis=1)) / (2 * self.sigma ** 2))

    def getQ(self, s, a):
        # Q = X.T.dot(Theta)
        return (self.rbfs(s)[a]).dot(self.theta_list[a])

    def select_action(self, s, eps):
        # e-greedyによる行動選択
        if np.random.rand() < eps:  # random
            action = np.random.randint(NUM_ACTION)
            return action
        else:
            qs = [self.getQ(s, i) for i in range(NUM_ACTION)]
            action = np.argmax(qs)
            # 最大値をとる行動が複数ある場合はさらにランダムに選択
            is_greedy_index = np.where(qs == action)[0]
            if len(is_greedy_index) > 1:
                action = np.random.choice(is_greedy_index)
            return action

    # パラメータの更新
    def update(self, s, a, reward, s_dash, a_dash):
        X = self.rbfs(s)
        Q_val = self.getQ(s, a)
        Q_val_dash = self.getQ(s_dash, a_dash)

        DELTA = reward + GAMMA * Q_val_dash - Q_val
        # パラメータの更新
        self.theta_list[a] = self.theta_list[a] + ALPHA * DELTA * X[a]

    def calc_Z(self):
        x = np.linspace(self.min_list[0], self.max_list[0], meshgrid)
        y = np.linspace(self.min_list[1], self.max_list[1], meshgrid)
        X, Y = np.meshgrid(x, y)
        Z = np.array([[[agent.getQ(np.array([i, j]), k) for i in x] for j in y] for k in range(NUM_ACTION)])
        return X, Y, Z


class NNAgent(BasicAgent):
    num_batch = 16
    hidden1 = 32
    hidden2 = 16
    min_list = env.observation_space.low  # 下限値 [-1.2 -0.07]
    max_list = env.observation_space.high  # 上限値 [-0.6, 0.07]

    def __init__(self, lr, h1, h2, in_dim, out_dim, eps_ini, eps_last):
        super().__init__(eps_ini, eps_last)
        self.done = False
        self.lr = lr  # learning rate
        self.h1 = h1  # 1st hidden layer
        self.h2 = h2  # 2nd hidden layer
        self.in_dim = (in_dim,)  # NNモデル生成のために入力変数を指定
        self.out_dim = out_dim  # NNモデル生成のために出力変数を指定
        self.eps_ini = eps_ini
        self.eps_last = eps_last
        self.alpha_ini = 0  # dummy # 他のクラスとの共通化のためダミー
        self.alpha_last = 0  # dummy
        self.model = self.build_model()  # モデルを生成

        optimizer = RMSprop(self.lr)
        self.model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

    def build_model(self):
        model = Sequential()
        model.add(Dense(self.h1, input_shape=self.in_dim, activation='relu'))
        model.add(Dense(self.h2, activation='relu'))
        model.add(Dense(self.out_dim, activation='linear', init='zero'))
        model.summary()
        return model

    def getQ(self, s, a):
        s = s.reshape(1, len(s))  # 2次元配列に整形
        q = self.model.predict(s)
        return self.model.predict(s)[0][a]

    def select_action(self, s, eps):
        # e-greedyによる行動選択
        if np.random.rand() < eps:  # random
            action = np.random.randint(NUM_ACTION)
            return action
        else:
            qs = self.model.predict(s.reshape(1, len(s)))[0]
            action = np.argmax(qs)
            is_greedy_index = np.where(qs == action)[0]
            if len(is_greedy_index) > 1:
                action = np.random.choice(is_greedy_index)
            return action

    def update(self, s, a, reward, s_dash, a_dash):
        # special reward for DQN
        reward = 0
        if self.done:
            if reward > -200:
                reward = +1
            else:
                reward = -1

        s = s.reshape(1, 2)
        s_dash = s_dash.reshape(1, 2)

        Q_val_dash = np.max(self.model.predict(s_dash), axis=1)
        targets = self.model.predict(s)
        target = reward + GAMMA * Q_val_dash * (self.done - 1.) * -1.  # means r when done else r + GAMMA * Q'
        targets[0, a] = target
        self.model.train_on_batch(s, targets)


class Result:
    def __init__(self):
        self.action_list = []
        self.reward_list = []



if __name__ == "__main__":
    # メイン関数

    # 1. Agentの選択
    agent_table = TableAgent(N=30, alpha_ini=0.5, alpha_last=0, eps_ini=0, eps_last=0)
    agent_linear = LinearFuncAgent(N=30, sigma=0.1, alpha_ini=0.2, alpha_last=0, eps_ini=0, eps_last=0)
    # agent_NN = NNAgent(lr=1.E-4, h1=320, h2=160, in_dim=NUM_STATE, out_dim=NUM_ACTION, eps_ini=0, eps_last=0)

    agent = agent_table  # 使用するエージェントを選択

    res = Result()  # 結果を格納するクラス

    # 学習
    for epi in range(int(num_episode)):

        # 行動履歴を格納するテーブルを初期化
        res.action_result = np.zeros(NUM_ACTION)
        tmp = 0  # 報酬積算用
        count = 0

        # greedy方策を徐々に確定的にしていく
        EPSILON = max(agent.eps_last, agent.eps_ini * (1 - epi * 1. / num_episode))
        ALPHA = max(agent.alpha_last, agent.alpha_ini * (1 - epi * 1. / num_episode))

        done = False
        # sを初期化
        s = env.reset()  # 環境をリセット
        agent.done = False
        # e-greedyによる行動選択
        a = agent.select_action(s, EPSILON)

        while (agent.done == False):
            if render:
                env.render()

            # 行動aをとり、r, s'を観測
            s_dash, reward, done, info, _ = env.step(a)
            agent.done = done
            # s'からe-greedyにより次の行動を決定
            a_dash = agent.select_action(s_dash, EPSILON)
            # 価値（パラメータ）の更新
            agent.update(s, a, reward, s_dash, a_dash)

            if agent.done:
                if count < 199:
                    print("SUCEED!!")

            a = a_dash
            s = s_dash

            # 行動履歴に追加
            res.action_result[a_dash] += 1

            tmp += reward
            count += 1

        res.reward_list.append(tmp)

        # print( agent.theta_list)
        print("epi: %d, eps: %.3f, alpha: %.3f, reward %d: " % (epi, EPSILON, ALPHA, tmp))

    env.close()