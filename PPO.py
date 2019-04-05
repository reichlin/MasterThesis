import math
import tensorflow as tf
import numpy as np
import random
import tensorflow_probability as tfp
import matplotlib.pyplot as plt


class Agent:

    def __init__(self, action_size, state_size, img_h, img_w, n_features, c1, c2, epochs, batch_size):

        self.epochs = epochs
        self.batch_size = batch_size
        self.n_features = n_features
        self.heat_map_size = int(((img_h - 6) / 2.) - 4 - 4)

        self.regularizer = None #tf.contrib.layers.l2_regularizer(scale=0.001)
        self.initializer = None

        # counters for wrinting summaries to tensorboard
        self.i = 0 # overall training
        self.i_p = 0 # perception pre-training
        self.i_t = 0 # transition pre-training
        self.update_r = 0 # reward
        self.update_s = 0 # score

        self.state_size = state_size
        self.action_size = action_size

        self.sess = tf.Session()

        with tf.variable_scope("model"):

            # Placeholders Model
            self.o_t = tf.placeholder(shape=[None, img_h, img_w, 3], dtype=tf.float32)
            self.o_t = self.o_t / 255.
            self.x_t1 = tf.placeholder(shape=[None, (2*self.n_features)], dtype=tf.float32)
            self.u_t1 = tf.placeholder(shape=[None, self.action_size], dtype=tf.float32)

            # Placeholder Pre-train
            self.o_reconstructed = tf.placeholder(shape=[None, 60, 60, 1], dtype=tf.float32)
            #self.o_reconstructed = self.o_reconstructed / 255.

            # Placeholders PPO
            self.state_arm = tf.placeholder(shape=[None, self.state_size], dtype=tf.float32)
            self.action = tf.placeholder(shape=[self.batch_size, self.action_size], dtype=tf.float32)
            self.V_targ = tf.placeholder(shape=[self.batch_size], dtype=tf.float32)
            self.advantage = tf.placeholder(shape=[self.batch_size], dtype=tf.float32)

            # Placeholders summaries
            self.reward = tf.placeholder(shape=(), dtype=tf.float32)
            self.score = tf.placeholder(shape=(), dtype=tf.float32)

            # Placeholders for Training
            self.lr = tf.placeholder(shape=(), dtype=tf.float32)
            self.lr_v = tf.placeholder(shape=(), dtype=tf.float32)
            self.epsilon = tf.placeholder(shape=(), dtype=tf.float32)
            self.sigma_val = tf.placeholder(shape=(), dtype=tf.float32)

            # constants
            self.n = tf.constant(self.action_size, dtype=tf.float32)
            self.c1 = tf.constant(c1)
            self.c2 = tf.constant(c2)
            self.pi_greco = tf.constant(math.pi)

            # Sigma matrix = diagonal with same value for each dimension

            self.pi_sigma = tf.ones(self.batch_size, tf.float32) * self.sigma_val

            # Define models

            self.o_tilde, self.o_hat, self.x_0, self.x_t, self.V, self.pi_mu = self.build_model("new")
            _, _, _, _, _, self.pi_old_mu = self.build_model("old")

            self.s_o_reconstructed = tf.summary.image("Scaled_image", self.o_reconstructed)

            # Pre-train perception
            self.loss_perception = tf.reduce_mean(tf.squared_difference(self.o_reconstructed, self.o_tilde))
            self.s_pretrain_p = tf.summary.scalar("Loss_pretrain_perception", self.loss_perception)
            self.s_o_tilde = tf.summary.image("reconstructed_perception", self.o_tilde)
            self.merge_p = tf.summary.merge([self.s_pretrain_p, self.s_o_tilde])
            self.optimizer_pretrain_p = tf.train.AdamOptimizer(self.lr)
            self.pretrain_p = self.optimizer_pretrain_p.minimize(self.loss_perception)

            # Pre-train transition
            self.loss_transition = tf.reduce_mean(tf.squared_difference(self.o_reconstructed, self.o_hat))
            self.s_pretrain_t = tf.summary.scalar("Loss_pretrain_transition", self.loss_transition)
            self.s_o_hat = tf.summary.image("reconstructed_transition", self.o_hat)
            self.merge_t = tf.summary.merge([self.s_pretrain_t, self.s_o_hat])
            self.optimizer_pretrain_t = tf.train.AdamOptimizer(self.lr)
            self.pretrain_t = self.optimizer_pretrain_t.minimize(self.loss_transition)

            # Compute Gaussian value of the action taken in log space

            self.mahalanobis_new = -0.5 * tf.reduce_sum(tf.pow(self.action - self.pi_mu, 2), axis=1) / self.pi_sigma
            self.norm_new = tf.pow(2.0 * self.pi_greco * self.pi_sigma, -(self.n/2.0))
            self.pi_sampled_log = tf.log(self.norm_new + 1e-10) + self.mahalanobis_new

            self.mahalanobis_old = -0.5 * tf.reduce_sum(tf.pow(self.action - self.pi_old_mu, 2), axis=1) / self.pi_sigma
            self.norm_old = tf.pow(2.0 * self.pi_greco * self.pi_sigma, -(self.n/2.0))
            self.pi_old_sampled_log = tf.log(self.norm_old + 1e-10) + self.mahalanobis_old

            # PPO Loss

            self.ratio = tf.exp(self.pi_sampled_log - tf.stop_gradient(self.pi_old_sampled_log))
            self.sur1 = tf.multiply(self.ratio, self.advantage)
            self.sur2 = tf.multiply(tf.clip_by_value(self.ratio, 1.0 - self.epsilon, 1.0 + self.epsilon), self.advantage)
            self.L_CLIP = tf.reduce_mean(tf.minimum(self.sur1, self.sur2))

            self.L_V = 0.5 * tf.reduce_mean(tf.squared_difference(self.V_targ, self.V))

            self.entropy = (self.n / 2.0) * (1.0 + tf.log(2.0 * self.pi_greco)) + 0.5 * tf.log(tf.reduce_prod(self.pi_sigma))

            self.loss = - self.L_CLIP + self.c1 * self.L_V - self.c2 * self.entropy

            # Training summaries

            self.s_pi = tf.summary.scalar('pi', tf.reduce_mean(tf.exp(self.pi_sampled_log)))
            self.s_ratio = tf.summary.scalar('Ratio', tf.reduce_mean(self.ratio))
            self.s_v = tf.summary.scalar('Loss_V', self.L_V)
            self.s_c = tf.summary.scalar('Loss_CLIP', -self.L_CLIP)
            self.s_e = tf.summary.scalar('Loss_entropy', -self.entropy)

            self.merge = tf.summary.merge([self.s_pi, self.s_ratio, self.s_v, self.s_c, self.s_e])

            self.s_r = tf.summary.scalar('Reward', self.reward)
            self.s_s = tf.summary.scalar('Score', self.score)

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
                                     filters=64,
                                     kernel_size=[7, 7],
                                     strides=(2, 2),
                                     activation=tf.nn.relu,
                                     padding="valid",
                                     kernel_initializer=self.initializer,
                                     kernel_regularizer=self.regularizer,
                                     name="conv1_p")

            conv1 = tf.layers.batch_normalization(inputs=conv1, training=True, name="batch_norm_conv1")

            conv2 = tf.layers.conv2d(inputs=conv1,
                                     filters=32,
                                     kernel_size=[5, 5],
                                     strides=(1, 1),
                                     activation=tf.nn.relu,
                                     padding="valid",
                                     kernel_initializer=self.initializer,
                                     kernel_regularizer=self.regularizer,
                                     name="conv2_p")

            conv2 = tf.layers.batch_normalization(inputs=conv2, training=True, name="batch_norm_conv2")

            conv3 = tf.layers.conv2d(inputs=conv2,
                                     filters=self.n_features,
                                     kernel_size=[5, 5],
                                     strides=(1, 1),
                                     activation=tf.nn.relu,
                                     padding="valid",
                                     kernel_initializer=self.initializer,
                                     kernel_regularizer=self.regularizer,
                                     name="conv3_p")

            conv3 = tf.layers.batch_normalization(inputs=conv3, training=True, name="batch_norm_conv3")

            # Spatial softmax --------------------------------------------------------------------------------------------------------------------------------------

            shape = tf.shape(conv3, name="shape_p1")

            temperature_p = tf.Variable(1., name="temperature_p/val")

            y = tf.exp(tf.div(conv3, temperature_p, name="div_p1"), name="exp_p1")

            y_sum = tf.reshape(tf.reduce_sum(y, axis=[1, 2], name="rs_p1"), [-1, self.n_features], name="reshape_p1")
            denominator = tf.reshape(tf.tile(y_sum, [1, shape[1] * shape[2]], name="tile_p1"), shape, name="reshape_p2")

            y = tf.div(y, denominator, name="div_p2")

            fi_p = tf.cast(tf.expand_dims(tf.reshape(tf.tile(tf.expand_dims(tf.tile(tf.range(0, self.heat_map_size, 1, name="range_p"),
                                                                                    [self.heat_map_size], name="tile_p2"),
                                                                            -1, name="expand_dim_p1"),
                                                             [1, self.n_features], name="tile_p3"),
                                                     [self.heat_map_size, self.heat_map_size, self.n_features], name="reshape_p3"),
                                          0, name="expand_dims_p2"),
                           tf.float32, name="cast_p1")

            fj_p = tf.transpose(fi_p, [0, 2, 1, 3], name="transpose_p")

            yi_p = tf.reduce_sum(tf.reshape(tf.multiply(y,
                                                        fi_p, name="mul_p1"),
                                            [-1, self.heat_map_size * self.heat_map_size, self.n_features], name="reshape_p4"),
                                 [1], name="rs_p2")

            yj_p = tf.reduce_sum(tf.reshape(tf.multiply(y,
                                                        fj_p, name="mul_p2"),
                                            [-1, self.heat_map_size * self.heat_map_size, self.n_features], name="reshape_p5"),
                                 [1], name="rs_p3")

            xt_ot = tf.concat([yi_p, yj_p], axis=1, name="concat_p1")

            # ------------------------------------------------------------------------------------------------------------------------------------------------------

            o_tilde = self.phi(xt_ot, False)

            # importance weights
            w_input = tf.concat([xt_ot, self.x_t1], -1, name="concat_w")

            alpha = tf.layers.dense(w_input,
                                    self.n_features,
                                    activation=tf.nn.sigmoid,
                                    kernel_initializer=self.initializer,
                                    kernel_regularizer=self.regularizer,
                                    name="dense_w")

            # Transition Model
            transition_input = tf.concat([self.x_t1, self.u_t1], -1, name="concat_t1")

            x_transition = tf.layers.dense(transition_input,
                                           (self.n_features * 2 * self.heat_map_size),
                                           activation=None,
                                           kernel_initializer=self.initializer,
                                           kernel_regularizer=self.regularizer,
                                           name="Tdense1")

            x_transition = tf.reshape(x_transition, [-1, (self.n_features * 2), self.heat_map_size], name="reshape_t1")

            temperature_t = tf.Variable(1., name="temperature_t/val")

            x_transition = tf.exp(tf.div(x_transition, temperature_t, name="div_t1"), name="exp_t1")

            x_transition = tf.div(x_transition,
                                  tf.tile(tf.expand_dims(tf.reduce_sum(x_transition,
                                                                       -1, name="rs_t1"),
                                                         -1, name="expand_dims_t1"),
                                          [1, 1, self.heat_map_size], name="tile_t1"), name="div_t2")

            fi_t = tf.cast(tf.tile(tf.expand_dims(tf.reshape(tf.tile(tf.range(0, self.heat_map_size, 1, name="range_t1"),
                                                                     [(self.n_features * 2)], name="tile_t2"),
                                                             [(self.n_features * 2), self.heat_map_size], name="reshape_t2"),
                                                  [0], name="expand_dims_t2"),
                                   [tf.shape(x_transition)[0], 1, 1], name="tile_t3"),
                           tf.float32, name="cast_t1")

            x_transition = tf.reduce_sum(tf.multiply(x_transition,
                                                     fi_t, name="mul_t1"),
                                         [-1], name="rs_t2")

            o_hat = self.phi(x_transition, True)

            # Observation Model: [w, w] * xt_ot + (1. - [w, w]) * x_transition

            w = tf.tile(alpha, [1, 2], name="tile_o")

            x_t = tf.add(tf.multiply(w, xt_ot, name="mul_o1"),
                         tf.multiply(tf.subtract(1., w, name="sub_o1"),
                                     x_transition, name="mul_o2"), name="add_o1")

            # Policy Model
            x = tf.concat([x_t, self.state_arm], -1, name="concat_policy")

            densev = tf.layers.dense(x,
                                     256,
                                     activation=tf.nn.relu,
                                     kernel_initializer=self.initializer,
                                     kernel_regularizer=self.regularizer,
                                     name="denseV_policy")

            dense1 = tf.layers.dense(x,
                                     256,
                                     activation=tf.nn.relu,
                                     kernel_initializer=self.initializer,
                                     kernel_regularizer=self.regularizer,
                                     name="dense1_policy")

            dense2 = tf.layers.dense(dense1,
                                     256,
                                     activation=tf.nn.relu,
                                     kernel_initializer=self.initializer,
                                     kernel_regularizer=self.regularizer,
                                     name="dense2_policy")

            dense3 = tf.layers.dense(dense2,
                                     32,
                                     activation=tf.nn.relu,
                                     kernel_initializer=self.initializer,
                                     kernel_regularizer=self.regularizer,
                                     name="dense3_policy")

            value = tf.squeeze(tf.layers.dense(densev,
                                               1,
                                               activation=None,
                                               kernel_initializer=self.initializer,
                                               kernel_regularizer=self.regularizer,
                                               name="value_policy"),
                               name="squeeze_policy")

            policy_mu = tf.layers.dense(dense3,
                                        self.action_size,
                                        activation=tf.nn.tanh,
                                        kernel_initializer=self.initializer,
                                        kernel_regularizer=self.regularizer,
                                        name="policy_mu")

        return o_tilde, o_hat, xt_ot, x_t, value, policy_mu

    def phi(self, features, init):

        image = tf.layers.dense(features,
                                3600,
                                activation=None,
                                kernel_initializer=self.initializer,
                                kernel_regularizer=self.regularizer,
                                name="dense_phi",
                                reuse=init)

        o = tf.reshape(image, [-1, 60, 60, 1], name="reshape_phi")
        return o

    def get_state_prior(self, img):

        prior = self.sess.run(self.x_0, feed_dict={self.o_t: np.expand_dims(img, 0)})
        return prior

    def get_state_value(self, img, xt1, ut1, arm):

        value = self.sess.run(self.V,
                              feed_dict={self.o_t: np.expand_dims(img, 0),
                                         self.x_t1: xt1,
                                         self.u_t1: np.expand_dims(ut1, 0),
                                         self.state_arm: np.expand_dims(arm, 0)})
        return value

    def get_action(self, img, xt1, ut1, arm, sigma):

        state, mu = self.sess.run((self.x_t, self.pi_old_mu),
                                  feed_dict={self.o_t: np.expand_dims(img, 0),
                                             self.x_t1: xt1,
                                             self.u_t1: np.expand_dims(ut1, 0),
                                             self.state_arm: np.expand_dims(arm, 0)})

        action = np.random.multivariate_normal(mu[0], np.diag(np.tile(sigma, np.size(mu[0]))))
        return state, action

    def pretrain_perception(self, img, img_recon, lr):

        summary, _ = self.sess.run((self.merge_p, self.pretrain_p),
                                   feed_dict={self.o_t: np.expand_dims(img, 0),
                                              self.o_reconstructed: np.expand_dims(img_recon, 0),
                                              self.lr: lr})

        self.train_writer.add_summary(summary, self.i_p)
        self.i_p += 1

    def pretrain_transition(self, img_recon, xt1, ut1, lr):

        summary, _ = self.sess.run((self.merge_t, self.pretrain_t),
                                   feed_dict={self.x_t1: xt1,
                                              self.u_t1: np.expand_dims(ut1, 0),
                                              self.o_reconstructed: np.expand_dims(img_recon, 0),
                                              self.lr: lr})

        self.train_writer.add_summary(summary, self.i_t)
        self.i_t += 1

    def normalize_advantages(self, advantages):

        mean = np.mean(advantages)
        std = np.std(advantages)
        return (advantages - mean) / (np.sqrt(std) + 1e-10)

    def fit(self, img, arm, actions, advantages, R, xt1, ut1, lr, lrv, epsilon, sigma):

        #advantages = self.normalize_advantages(advantages)

        for e in range(self.epochs):

            n_batches = int(np.size(arm, 0) / self.batch_size)
            idx = np.random.permutation(int(np.size(arm, 0)))

            img = img[idx]
            arm = arm[idx]
            actions = actions[idx]
            advantages = advantages[idx]
            R = R[idx]
            xt1 = xt1[idx]
            ut1 = ut1[idx]

            for b in range(n_batches):

                summary, _, _ = self.sess.run((self.merge, self.train_ppo, self.train_ppo_v),
                                              feed_dict={self.o_t: img[b * self.batch_size:(b + 1) * self.batch_size],
                                                         self.state_arm: arm[b * self.batch_size:(b + 1) * self.batch_size],
                                                         self.advantage: advantages[b * self.batch_size:(b + 1) * self.batch_size],
                                                         self.action: actions[b * self.batch_size:(b + 1) * self.batch_size],
                                                         self.V_targ: R[b * self.batch_size:(b + 1) * self.batch_size],
                                                         self.x_t1: xt1[b * self.batch_size:(b + 1) * self.batch_size],
                                                         self.u_t1: ut1[b * self.batch_size:(b + 1) * self.batch_size],
                                                         self.lr: lr,
                                                         self.lr_v: lrv,
                                                         self.epsilon: epsilon,
                                                         self.sigma_val: sigma})

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

    def write_score(self, score):

        s = self.sess.run(self.s_s, feed_dict={self.score: score})
        self.train_writer.add_summary(s, self.update_s)
        self.update_s += 1

    def save_model(self):

        saver = tf.train.Saver()

        try:
            saver.save(self.sess, "/Users/alfredoreichlin/PycharmProjects/thesis/venv/models/model.ckpt")
            print("model saved successfully")
        except Exception:
            pass
