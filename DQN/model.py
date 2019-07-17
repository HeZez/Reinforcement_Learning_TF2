import os
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp


print("TensorFlow Ver: ", tf.__version__)
print("Eager Execution:", tf.executing_eagerly())


class Q_model(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super().__init__('Q_Model')
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = 0.2
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(64, input_shape=(None, self.state_dim), activation='relu'))
        self.model.add(tf.keras.layers.Dense(32, activation='relu'))
        self.model.add(tf.keras.layers.Dense(self.action_dim))

    @tf.function
    def call(self, inputs, *args, **kwargs):
        x = tf.convert_to_tensor(inputs)
        Q_Value = self.model(x)
        return Q_Value

    def choose_action(self, obs):
        Q_Value = self.predict(obs)

        if np.random.random_sample(1) < self.epsilon:
            # 随机选择动作
            action = np.random.randint(0, self.action_dim, 1)
        else:
            # 贪婪策略选择动作
            action = np.argmax(Q_Value)
            action = [action]
        return action


