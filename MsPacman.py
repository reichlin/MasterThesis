import gym
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import PPO_discrete as PPO

tf.reset_default_graph()


def alpha_scheduler(step, STEPS):

    x = (1. * step / STEPS)
    a = - x + 1
    return a


K = 2
T = 16
STEPS = int(128 / (K*T))

EPOCHS = 2
gamma = 0.99
gae = 0.95
batch_size = 8
c1 = 1.0
c2 = 0.01
lr_0 = 2.5 * 1e-4
lr_V_0 = 1.0 * 1e-6
epsilon_0 = 0.1

envs = []
done = []
for i in range(K):
    env = gym.make('MsPacman-v0')
    envs.append(env)
    done.append(True)

action_size = envs[0].action_space.n
img_h = 210
img_w = 160
n_channels = 1

agent = PPO.Agent(action_size, img_h, img_w, n_channels, c1, c2, EPOCHS, batch_size)

# create memory buffer
states = np.zeros((K, T, img_h, img_w, n_channels))
actions = np.zeros((K, T))
cumulative_reward = np.zeros((K, T))
advantages = np.zeros((K, T))

greedy = 1.0

imgs = np.zeros((K, img_h, img_w, n_channels))

for step in range(STEPS):

    if step % (STEPS/100) == 0:
        print(str(((step*100)/STEPS)) + "%")

    sum_reward = 0

    alpha = alpha_scheduler(step, STEPS)
    lr = lr_0 * alpha
    lr_V = lr_V_0 * alpha
    epsilon = epsilon_0 * alpha

    for t in range(T):

        for k in range(K):

            if done[k]:
                done[k] = False
                img = envs[k].reset()
                imgs[k,:,:,0] = (1. * img[:,:,0] + 1. * img[:,:,1] + 1. * img[:,:,2]) / (3. * 255.)

            if np.random.uniform() < greedy:
                pi = agent.get_action(imgs[k])
                action = int(np.random.choice(action_size, 1, p=np.squeeze(pi)))
            else:
                action = env.action_space.sample()

            img, reward, done[k], info = envs[k].step(action)
            next_img = (1. * img[:, :, 0] + 1. * img[:, :, 1] + 1. * img[:, :, 2]) / (3. * 255.)

            Vt = agent.get_state_value(imgs[k])
            Vt1 = agent.get_state_value(np.expand_dims(next_img, -1))
            delta = reward + gamma * Vt1 - Vt

            states[k, t] = imgs[k]
            actions[k, t] = action
            cumulative_reward[k, t] = reward
            advantages[k, t] = delta

            imgs[k,:,:,0] = next_img
            sum_reward += reward

    for kk in range(K):
        for tau in range(T-2, -1, -1):
            advantages[kk, tau] += (gamma * gae) * advantages[kk, tau + 1]
            cumulative_reward[kk, tau] += gamma * cumulative_reward[kk, tau + 1]

    agent.fit(states.reshape((K*T, img_h, img_w, n_channels)),
              actions.reshape((K*T)),
              advantages.reshape((K*T)),
              cumulative_reward.reshape((K*T)),
              lr,
              lr_V,
              epsilon)

    agent.update_old_policy()

    states = np.zeros((K, T, img_h, img_w, n_channels))
    actions = np.zeros((K, T))
    cumulative_reward = np.zeros((K, T))
    advantages = np.zeros((K, T))

    agent.write_reward(sum_reward/K)

agent.save_model()
