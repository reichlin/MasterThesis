import math
import tensorflow as tf
import numpy as np
import random


class Agent:

    def __init__(self, action_size, img_h, img_w, n_channels, c1, epochs, batch_size):

        self.epochs = epochs
        self.batch_size = batch_size

        self.regularizer = None #tf.contrib.layers.l2_regularizer(scale=0.001)
        self.initializer = None

        # counters for wrinting summaries to tensorboard
        self.i = 0 # overall training
        self.update_r = 0 # reward

        self.action_size = action_size

        self.sess = tf.Session()

        with tf.variable_scope("model"):

            # Placeholders Model
            self.o_t = tf.placeholder(shape=[None, img_h, img_w, n_channels], dtype=tf.float32)
            #self.o_t = self.o_t / 255.

            # Placeholders PPO
            self.action = tf.placeholder(shape=[self.batch_size], dtype=tf.int32)
            self.V_targ = tf.placeholder(shape=[self.batch_size], dtype=tf.float32)
            self.advantage = tf.placeholder(shape=[self.batch_size], dtype=tf.float32)

            # Placeholders summaries
            self.reward = tf.placeholder(shape=(), dtype=tf.float32)

            # Placeholders for Training
            self.lr = tf.placeholder(shape=(), dtype=tf.float32)
            self.lr_v = tf.placeholder(shape=(), dtype=tf.float32)
            self.epsilon = tf.placeholder(shape=(), dtype=tf.float32)
            self.c2 = tf.placeholder(shape=(), dtype=tf.float32)

            # constants
            self.n = tf.constant(self.action_size, dtype=tf.float32)
            self.c1 = tf.constant(c1)
            self.pi_greco = tf.constant(math.pi)

            # Define models

            self.V, self.pi = self.build_model("new")
            _, self.pi_old = self.build_model("old")

            # Compute Probability of the action taken in log space

            self.action_taken_one_hot = tf.one_hot(self.action, self.action_size)

            self.pi_sampled_log = tf.log(tf.reduce_sum(self.pi * self.action_taken_one_hot, -1) + 1e-5)
            self.pi_old_sampled_log = tf.log(tf.reduce_sum(self.pi_old * self.action_taken_one_hot, -1) + 1e-5)

            # PPO Loss

            self.ratio = tf.exp(self.pi_sampled_log - tf.stop_gradient(self.pi_old_sampled_log))
            self.sur1 = tf.multiply(self.ratio, self.advantage)
            self.sur2 = tf.multiply(tf.clip_by_value(self.ratio, 1.0 - self.epsilon, 1.0 + self.epsilon), self.advantage)
            self.L_CLIP = tf.reduce_mean(tf.minimum(self.sur1, self.sur2))

            self.L_V = 0.5 * tf.reduce_mean(tf.squared_difference(self.V_targ, self.V))

            self.entropy = - tf.reduce_sum(self.pi * tf.log(self.pi))

            self.loss = - self.L_CLIP + self.c1 * self.L_V - self.c2 * self.entropy

            # Training summaries

            self.s_pi = tf.summary.scalar('pi', tf.reduce_mean(tf.exp(self.pi_sampled_log)))
            self.s_ratio = tf.summary.scalar('Ratio', tf.reduce_mean(self.ratio))
            self.s_v = tf.summary.scalar('Loss_V', self.L_V)
            self.s_c = tf.summary.scalar('Loss_CLIP', -self.L_CLIP)
            self.s_e = tf.summary.scalar('Loss_entropy', -self.entropy)

            self.merge = tf.summary.merge([self.s_pi, self.s_ratio, self.s_v, self.s_c, self.s_e])

            self.s_r = tf.summary.scalar('Reward', self.reward)

            # Optimization steps

            self.optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_ppo = self.optimizer.minimize(self.loss)
            self.optimizer_v = tf.train.AdamOptimizer(self.lr_v)
            self.train_ppo_v = self.optimizer_v.minimize(self.L_V)

        with tf.variable_scope("assign"):

            self.assign_arr = []
            self.col_dict = {}
            self.col1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/new')
            for i in range(len(self.col1)):
                self.col_dict[self.col1[i].name.split('/')[-2] + "/" + self.col1[i].name.split('/')[-1]] = self.col1[i]

            self.col2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/old')
            for i in range(len(self.col2)):
                self.node_name = self.col2[i].name.split('/')[-2] + "/" + self.col2[i].name.split('/')[-1]
                self.assign0 = self.col2[i].assign(self.col_dict[self.node_name])
                self.assign_arr.append(self.assign0)

        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        self.train_writer = tf.summary.FileWriter('train/', self.sess.graph)

    def build_model(self, name):

        with tf.variable_scope(name):

            # Perception Model

            conv1 = tf.layers.conv2d(inputs=self.o_t,
                                     filters=16,
                                     kernel_size=[8, 8],
                                     strides=(4, 4),
                                     activation=tf.nn.relu,
                                     padding="valid",
                                     kernel_initializer=self.initializer,
                                     kernel_regularizer=self.regularizer,
                                     name="conv1")

            #conv1 = tf.layers.batch_normalization(inputs=conv1, training=True, name="batch_norm_conv1")

            conv2 = tf.layers.conv2d(inputs=conv1,
                                     filters=32,
                                     kernel_size=[4, 4],
                                     strides=(2, 2),
                                     activation=tf.nn.relu,
                                     padding="valid",
                                     kernel_initializer=self.initializer,
                                     kernel_regularizer=self.regularizer,
                                     name="conv2")

            flat = tf.layers.flatten(conv2, name="flatten")

            dense1 = tf.layers.dense(flat,
                                     256,
                                     activation=tf.nn.relu,
                                     kernel_initializer=self.initializer,
                                     kernel_regularizer=self.regularizer,
                                     name="dense1")

            value = tf.squeeze(tf.layers.dense(dense1,
                                               1,
                                               activation=None,
                                               kernel_initializer=self.initializer,
                                               kernel_regularizer=self.regularizer,
                                               name="value_policy"),
                               name="squeeze_policy")

            policy_mu = tf.layers.dense(dense1,
                                        self.action_size,
                                        activation=tf.nn.softmax,
                                        kernel_initializer=self.initializer,
                                        kernel_regularizer=self.regularizer,
                                        name="policy_mu")

        return value, policy_mu

    def get_state_value(self, img):

        value = self.sess.run(self.V, feed_dict={self.o_t: np.expand_dims(img, 0)})
        return value

    def get_action(self, img):

        pi = self.sess.run(self.pi_old, feed_dict={self.o_t: np.expand_dims(img, 0)})

        return pi

    def normalize_advantages(self, advantages):

        mean = np.mean(advantages)
        std = np.std(advantages)
        return (advantages - mean) / (np.sqrt(std) + 1e-10)

    def fit(self, img, actions, advantages, R, lr, lrv, epsilon, c2):

        #advantages = self.normalize_advantages(advantages)

        for e in range(self.epochs):

            n_batches = int(np.size(img, 0) / self.batch_size)
            idx = np.random.permutation(int(np.size(img, 0)))

            img = img[idx]
            actions = actions[idx]
            advantages = advantages[idx]
            R = R[idx]

            for b in range(n_batches):

                summary, _, _ = self.sess.run((self.merge, self.train_ppo, self.train_ppo_v),
                                              feed_dict={self.o_t: img[b * self.batch_size:(b + 1) * self.batch_size],
                                                         self.advantage: advantages[b * self.batch_size:(b + 1) * self.batch_size],
                                                         self.action: actions[b * self.batch_size:(b + 1) * self.batch_size],
                                                         self.V_targ: R[b * self.batch_size:(b + 1) * self.batch_size],
                                                         self.lr: lr,
                                                         self.lr_v: lrv,
                                                         self.epsilon: epsilon,
                                                         self.c2: c2})

                self.train_writer.add_summary(summary, self.i)
                self.i += 1

        return

    def update_old_policy(self):

        self.sess.run(self.assign_arr)
        return

    def write_reward(self, reward):

        r = self.sess.run(self.s_r, feed_dict={self.reward: reward})
        self.train_writer.add_summary(r, self.update_r)
        self.update_r += 1

    def save_model(self):

        saver = tf.train.Saver()

        try:
            saver.save(self.sess, "/Users/alfredoreichlin/PycharmProjects/thesis/venv/models/MsPacmanModel.ckpt")
            print("model saved successfully")
        except Exception:
            pass
