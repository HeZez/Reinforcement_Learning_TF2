import numpy as np
from DQN import DQN
import matplotlib.pyplot as plt
import math
import gym
from memory import Memory
import tensorflow as tf


env = gym.make('CartPole-v0')
# env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

print(env.action_space)
print(env.observation_space.shape[0])
print(env.observation_space.high)
print(env.observation_space.low)


def train(DQN, Memory):
    RENDER = False
    count = 0
    all_ep_reard = []
    loss_all = []
    for episode in range(200):
        obs = env.reset()
        num_plays_per_episode = 0
        epsode_reward = 0
        while True:
            # if RENDER:
            #     env.render()
            # obs = obs[np.newaxis, :]
            # action = DQN.DQN_eval.choose_action(obs)
            action = DQN.DQN_eval.choose_action(obs[np.newaxis, :])
            obs_, reward, done, _ = env.step(action[0])
            epsode_reward += reward
            # reward是一个float 包装成数组
            reward = [reward]
            Memory.save(obs, action, reward, obs_, count)
            count += 1
            num_plays_per_episode += 1
            if count > 200:
                data = Memory.get_data(32, count)
                loss = DQN.update(data)
                if len(loss_all) > 1:
                    loss_all.append(0.1 * loss + 0.9 * loss_all[-1])
                else:
                    loss_all.append(loss)
                if count % 200 == 0:
                    DQN.update_target()
                    # a = 1
            if num_plays_per_episode == 199:
                break
            obs = obs_

        if epsode_reward > 100.0:
            RENDER = True
        if 'running_reward' not in globals():
            running_reward = epsode_reward
        else:
            running_reward = running_reward * 0.9 + epsode_reward * 0.1
        all_ep_reard.append(running_reward)

        print("episode:", episode, "  reward:", epsode_reward)
    plt.plot(np.arange(len(loss_all)), loss_all)
    plt.show()


if __name__ == '__main__':
    Memory = Memory(1000, env.observation_space.shape[0], 1)
    momentum = tf.constant(0.1, dtype=tf.float32)
    DQN = DQN(env.observation_space.shape[0], 2, lr=0.005, momentum=momentum)
    DQN.init_weights(env.reset())
    train(DQN, Memory)
