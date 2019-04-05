import gym
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import cv2
import PPO

tf.reset_default_graph()


def lr_schedule(lr_max, game, EPISODES):

    lr = lr_max * np.exp(- 4. * ((game * 1.) / EPISODES))
    return lr


def epsilon_schedule(epsilon_max, epsilon_min, game, EPISODES):

    epsilon = (epsilon_min - epsilon_max) * ((game * 1.) / EPISODES) + epsilon_max
    return epsilon


def sigma_schedule(max_sigma, min_sigma, game, EPISODES):

    sigma = (min_sigma - max_sigma) * ((game * 1.) / EPISODES) + max_sigma
    return sigma


''' CONSTANTS '''

EPISODES = 2
EPOCHS = 2
K = 4
T = 50
gamma = 0.0
gae = 0.0
batch_size = 8
lr_max = 1.0 * 1e-3
lr_min = 1.0 * 1e-6
lr_V_max = 1.0 * 1e-3
c1 = 1.0
c2 = 0.0
epsilon_max = 0.1
epsilon_min = 0.0001
min_sigma = 0.01
max_sigma = 1.0

img_h = 240
img_w = 240
n_features = 16

STEPS_PERCEPTION = 3000
STEPS_TRANSITION = 3000


''' REAL PROGRAM '''

env = gym.make('FetchPush-v2')

action_size = env.action_space.shape[0]
state_size = env.observation_space.spaces['observation'].shape[0]

print("start")

agent = PPO.Agent(action_size, state_size, img_h, img_w, n_features, c1, c2, EPOCHS, batch_size)

print("agent created")

# pre-train perception

obs = env.reset()
img = env.render('rgb_array')
done = False

for step in range(STEPS_PERCEPTION):

    lr = lr_schedule(1e-3, step, STEPS_PERCEPTION)

    if step % (STEPS_PERCEPTION/100) == 0:
        print(str(((step*100)/STEPS_PERCEPTION)) + "% of pre-training perception")

    if done:
        obs = env.reset()
        img = env.render('rgb_array')
        done = False

    # reconstruct image
    img_scaled = cv2.resize(img, (60, 60))

    red = np.clip(img_scaled[:, :, 0] - 200., 0., 1.)
    green = np.clip(-img_scaled[:, :, 1] - 200., 0., 1.)
    blue = np.clip(-img_scaled[:, :, 2] - 200., 0., 1.)
    img_red = red * green * blue

    img_recon = np.expand_dims(img_red, -1)

    agent.pretrain_perception(img, img_recon, lr)

    action = env.action_space.sample()

    _, _, done, _ = env.step(action)
    img = env.render('rgb_array')

print("perception pre-training done")

# pre-train transition

obs = env.reset()
img = env.render('rgb_array')
x_t1 = agent.get_state_prior(img)
u_t1 = np.zeros(action_size)
done = False

for step in range(STEPS_TRANSITION):

    lr = lr_schedule(1e-3, step, STEPS_TRANSITION)

    if step % (STEPS_TRANSITION / 100) == 0:
        print(str(((step * 100) / STEPS_TRANSITION)) + "% of pre-training perception")

    if done:
        obs = env.reset()
        img = env.render('rgb_array')
        x_t1 = agent.get_state_prior(img)
        u_t1 = np.zeros(action_size)
        done = False

    action = env.action_space.sample()

    _, _, done, _ = env.step(action)
    next_img = env.render('rgb_array')

    # reconstruct image
    img_scaled = cv2.resize(next_img, (60, 60))
    red = np.clip(img_scaled[:, :, 0] - 200., 0, 1)
    green = np.clip(-img_scaled[:, :, 1] - 200., 0, 1)
    blue = np.clip(-img_scaled[:, :, 2] - 200., 0, 1)
    img_red = red * green * blue
    img_recon = np.expand_dims(img_red, -1)

    if np.sum(img_recon) > 0:
        agent.pretrain_transition(img_recon, x_t1, u_t1, lr)

    x_t1 = agent.get_state_prior(img)
    u_t1 = action

print("transition pre-training done")

# TODO: change greedy policy
greedy = 0.0

# create memory buffer
images = np.zeros((K, T, img_h, img_w, 3))
states = np.zeros((K, T, state_size))
actions = np.zeros((K, T, action_size))
cumulative_reward = np.zeros((K, T))
advantages = np.zeros((K, T))
past_states = np.zeros((K, T, (2*n_features)))
past_actions = np.zeros((K, T, action_size))

for game in range(EPISODES):

    if game % (EPISODES/100) == 0:
        print(str(((game*100)/EPISODES)) + "% of training")

    # TODO: riguardare gli scheduler
    lr = lr_schedule(lr_max, game, EPISODES)
    lr_V = lr_schedule(lr_V_max, game, EPISODES)
    epsilon = epsilon_schedule(epsilon_max, epsilon_min, game, EPISODES)
    sigma = sigma_schedule(max_sigma, min_sigma, game, EPISODES)

    sum_reward = 0
    sum_score = 0

    for k in range(K):

        obs = env.reset()
        img = env.render('rgb_array')
        arm_state = obs['observation']
        x_t1 = agent.get_state_prior(img)
        u_t1 = np.zeros(action_size)
        done = False

        t = 0

        while not done:

            if np.random.uniform() < greedy:
                x_t, action = agent.get_action(img, x_t1, u_t1, arm_state, sigma)
            else:
                action = env.action_space.sample()
                x_t = agent.get_state_prior(img)

            next_obs, reward, done, info = env.step(action)
            next_arm_state = next_obs['observation']
            next_img = env.render('rgb_array')

            # Compute Reward ---------------------------------------------------------------------------------------------------------------------------------------

            gripper = next_obs['observation'][0:3]
            red_ball = next_obs['desired_goal1']
            green_ball = next_obs['desired_goal2']
            cube = next_obs['achieved_goal']

            dist_gripper_red = np.sqrt((gripper[0] - red_ball[0]) ** 2 +
                                       (gripper[1] - red_ball[1]) ** 2 +
                                       (gripper[2] - red_ball[2]) ** 2)

            reward = - dist_gripper_red

            if dist_gripper_red < 0.05:
                score = 0.
            else:
                score = -1.

            # ------------------------------------------------------------------------------------------------------------------------------------------------------

            V_t = agent.get_state_value(img, x_t1, u_t1, arm_state)
            V_t1 = agent.get_state_value(next_img, x_t, action, next_arm_state)
            delta = reward + gamma * V_t1 - V_t

            images[k, t] = img
            states[k, t] = arm_state
            actions[k, t] = action
            advantages[k, t] = delta
            cumulative_reward[k, t] = reward
            past_states[k, t] = x_t1[0]
            past_actions[k, t] = u_t1

            img = next_img
            arm_state = next_arm_state
            x_t1 = x_t
            u_t1 = action

            t += 1

            sum_reward += reward
            sum_score += score

        for tau in range(T-2, -1, -1):
            advantages[k, tau] += (gamma * gae) * advantages[k, tau + 1]
            cumulative_reward[k, tau] += gamma * cumulative_reward[k, tau + 1]

    agent.fit(images.reshape((K*T, img_h, img_w, 3)),
              states.reshape((K*T, state_size)),
              actions.reshape((K*T, action_size)),
              advantages.reshape((K*T)),
              cumulative_reward.reshape((K*T)),
              past_states.reshape((K*T, (2*n_features))),
              past_actions.reshape((K*T, action_size)),
              lr,
              lr_V,
              epsilon,
              sigma)

    agent.update_old_policy()

    images = np.zeros((K, T, img_h, img_w, 3))
    states = np.zeros((K, T, state_size))
    actions = np.zeros((K, T, action_size))
    cumulative_reward = np.zeros((K, T))
    advantages = np.zeros((K, T))
    past_states = np.zeros((K, T, (2 * n_features)))
    past_actions = np.zeros((K, T, action_size))

    agent.write_reward(sum_reward/K)
    agent.write_score(sum_score/K)

# agent.save_model()

# TODO: add gif creation

# Test
counter = 0
for counter in range(50):
    done = False
    obs = env.reset()
    arm_state = obs['observation']
    img = env.render('rgb_array')
    x_t1 = agent.get_state_prior(img)
    u_t1 = np.zeros(action_size)
    final_score = 0
    while not done:
        x_t, action = agent.get_action(img, x_t1, u_t1, arm_state, sigma)
        next_obs, reward, done, info = env.step(action)
        next_arm_state = next_obs['observation']
        next_img = env.render('rgb_array')

        gripper = next_obs['observation'][0:3]
        red_ball = next_obs['desired_goal1']
        green_ball = next_obs['desired_goal2']
        cube = next_obs['achieved_goal']

        dist_gripper_red = np.sqrt((gripper[0] - red_ball[0]) ** 2 +
                                   (gripper[1] - red_ball[1]) ** 2 +
                                   (gripper[2] - red_ball[2]) ** 2)

        if dist_gripper_red < 0.05:
            score = 0.
        else:
            score = -1.

        img = next_img
        arm_state = next_arm_state
        x_t1 = x_t
        u_t1 = action

        final_score += score

    print("final score " + str(final_score))











