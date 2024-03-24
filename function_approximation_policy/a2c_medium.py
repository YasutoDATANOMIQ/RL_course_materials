import numpy as np
import tensorflow as tf
import gym
import tensorflow_probability as tfp
import tensorflow.keras.losses as kls

class Critic(tf.keras.Model):
    def __init__(self, model_type='dense'):
        super().__init__()

        self.model_type = model_type

        if self.model_type == 'dense':
            self.d1 = tf.keras.layers.Dense(128, activation='relu')
            self.d2 = tf.keras.layers.Dense(32,activation='relu')
            self.v = tf.keras.layers.Dense(1, activation=None)

    def call(self, input_data):
        if self.model_type == 'dense':
            x = self.d1(input_data)
            x = self.d2(x)
            v = self.v(x)
        return v

class Actor(tf.keras.Model):
    def __init__(self, action_size, model_type='dense'):
        super().__init__()

        self.model_type=model_type

        if self.model_type == 'dense':
            self.d1 = tf.keras.layers.Dense(128, activation='relu')
            self.d2 = tf.keras.layers.Dense(32,activation='relu')
            self.a = tf.keras.layers.Dense(action_size, activation='softmax')

    def call(self, input_data):
        if self.model_type == 'dense':
            x = self.d1(input_data)
            x = self.d2(x)
            a = self.a(x)
        return a

class A2CAgent():
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.a_opt = tf.keras.optimizers.RMSprop(learning_rate=7e-3)
        self.c_opt = tf.keras.optimizers.RMSprop(learning_rate=7e-3)
        self.actor = Actor()
        self.critic = Critic()

    def act(self, state):
        prob = self.actor(np.array([state]))
        prob = prob.numpy()
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])

    def actor_loss(self, probs, actions, td):

        probability = []
        log_probability = []
        for pb, a in zip(probs, actions):
            dist = tfp.distributions.Categorical(probs=pb, dtype=tf.float32)
            log_prob = dist.log_prob(a)
            prob = dist.prob(a)
            probability.append(prob)
            log_probability.append(log_prob)

        p_loss = []
        e_loss = []
        td = td.numpy()
        # print(td)
        for pb, t, lpb in zip(probability, td, log_probability):
            t = tf.constant(t)
            policy_loss = tf.math.multiply(lpb, t)
            entropy_loss = tf.math.negative(tf.math.multiply(pb, lpb))
            p_loss.append(policy_loss)
            e_loss.append(entropy_loss)
        p_loss = tf.stack(p_loss)
        e_loss = tf.stack(e_loss)
        p_loss = tf.reduce_mean(p_loss)
        e_loss = tf.reduce_mean(e_loss)


        loss = -p_loss - 0.0001 * e_loss
        # print(loss)
        return loss

    def learn(self, states, actions, discnt_rewards):
        discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),))

        with tf.GradientTape() as tape_actor, tf.GradientTape() as tape_critic:
            print('Checkpoint gradient tape')
            p = self.actor(states, training=True)
            v = self.critic(states, training=True)
            v = tf.reshape(v, (len(v),))
            td = tf.math.subtract(discnt_rewards, v)
            a_loss = self.actor_loss(p, actions, td)
            c_loss = 0.5 * kls.mean_squared_error(discnt_rewards, v)
        grads_actor = tape_actor.gradient(a_loss, self.actor.trainable_variables)
        grads_critic = tape_critic.gradient(c_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(grads_actor, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(grads_critic, self.critic.trainable_variables))
        return a_loss, c_loss

class A2CTrainer():
    def __init__(self, env):
        self.a2c_agent = A2CAgent()
        self.env = env

    def preprocess(self, states, actions, rewards, gamma):
        discnt_rewards = []
        sum_reward = 0
        rewards.reverse()
        for r in rewards:
            sum_reward = r + gamma * sum_reward
            discnt_rewards.append(sum_reward)
        discnt_rewards.reverse()
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        discnt_rewards = np.array(discnt_rewards, dtype=np.float32)

        return states, actions, discnt_rewards

    def train(self, total_episode_num=100, epsilon=0.1):

        ep_reward = []
        total_avgr = []

        for episode_cnt in range(total_episode_num):
            done = False
            state = self.env.reset()

            if type(state) == tuple:
                state = state[0]

            total_reward = 0
            all_aloss = []
            all_closs = []
            rewards = []
            states = []
            actions = []

            while not done:

                action = self.a2c_agent.act(state)
                next_state, reward, done, _, _ = self.env.step(action)
                rewards.append(reward)
                states.append(state)
                # actions.append(tf.one_hot(action, 2, dtype=tf.int32).numpy().tolist())
                actions.append(action)
                state = next_state
                total_reward += reward

                if done:
                    ep_reward.append(total_reward)
                    avg_reward = np.mean(ep_reward[-100:])
                    total_avgr.append(avg_reward)
                    print("Total reward after {} steps is {} and avg reward is {}".format(episode_cnt, total_reward, avg_reward))
                    states, actions, discnt_rewards = self.preprocess(states, actions, rewards, self.a2c_agent.gamma)

                    al, cl = self.a2c_agent.learn(states, actions, discnt_rewards)

if __name__ == "__main__":
    # LunarLander-v2
    env= gym.make("CartPole-v0")
    # env = gym.make("Acrobot-v1")
    trainer = A2CTrainer(env)

    trainer.train()

