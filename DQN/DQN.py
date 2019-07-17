import tensorflow as tf
import numpy as np
from model import Q_model


class DQN(object):
    def __init__(self, state_dim, action_dim, gamma=0.95, lr=0.001, momentum=0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.DQN_eval = Q_model(state_dim, action_dim)
        self.DQN_target = Q_model(state_dim, action_dim)
        # self.DQN_target.set_weights(self.DQN_eval.get_weights())
        self.gamma = gamma
        self.lr = lr
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.momentum = momentum
        self.batch_size = 32

    def Q_loss(self, q_eval, q_target):
        loss = tf.reduce_mean(tf.square(q_eval-q_target))
        return loss

    def update(self, batch_memory):
        obs = batch_memory[:, :self.state_dim]
        obs_ = batch_memory[:, -self.state_dim:]
        reward = batch_memory[:, self.state_dim+1]
        action = batch_memory[:, self.state_dim].astype(int)
        # 更新eval net
        loss = self.train_one_step(obs, action, reward, obs_)
        return loss

    # @tf.function
    def train_one_step(self, obs, action, reward, obs_):
        with tf.GradientTape() as tape:
            # 求Q(s',a')
            q_next = self.DQN_target(obs_)
            # q_next = self.DQN_eval(obs_)
            # 求Q(s,a)
            q_now = self.DQN_eval(obs)
            # print(q_next)
            # 构造Q_target
            q_target = q_now
            q_target = q_target.numpy()
            batct_num = np.arange(self.batch_size, dtype=np.int32)
            q_target[batct_num, action] = reward + self.gamma * np.max(q_next.numpy(), axis=1)
            # 得到Loss
            loss = self.Q_loss(q_now, q_target)

        grads = tape.gradient(loss, self.DQN_eval.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.DQN_eval.trainable_variables))
        # self.optimizer.minimize(loss=loss, var_list=self.DQN_eval.trainable_variables)
        return loss

    # 初始化网络权重
    def init_weights(self, obs):
        self.DQN_eval(obs[np.newaxis, :])
        self.DQN_target(obs[np.newaxis, :])
        self.DQN_target.set_weights(self.DQN_eval.get_weights())

    # 更新target网络
    def update_target(self):
        self.DQN_target.set_weights(self.DQN_eval.get_weights())



