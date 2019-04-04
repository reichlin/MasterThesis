import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import tensorflow as tf

tf.reset_default_graph()

initializer = tf.initializers.he_normal()
batch = 16
epochs = 5

x = tf.placeholder(shape=[None, 240, 240, 3], dtype=tf.float32)
label = tf.placeholder(shape=[batch, 60, 60, 1], dtype=tf.float32)
training = tf.placeholder(shape=(), dtype=tf.bool)

x = x / 255.

tf.summary.image("x", x, 1)

conv1 = tf.layers.conv2d(inputs=x,
                         filters=64,
                         kernel_size=[7, 7],
                         strides=(2, 2),
                         activation=tf.nn.relu,
                         padding="valid",
                         kernel_initializer=initializer,
                         name="conv1")

conv1 = tf.layers.batch_normalization(inputs=conv1, training=training)

conv2 = tf.layers.conv2d(inputs=conv1,
                         filters=32,
                         kernel_size=[5, 5],
                         strides=(1, 1),
                         activation=tf.nn.relu,
                         padding="valid",
                         kernel_initializer=initializer,
                         name="conv2")

conv2 = tf.layers.batch_normalization(inputs=conv2, training=training)

conv3 = tf.layers.conv2d(inputs=conv2,
                         filters=16,
                         kernel_size=[5, 5],
                         strides=(1, 1),
                         activation=tf.nn.relu,
                         padding="valid",
                         kernel_initializer=initializer,
                         name="conv3")

conv3 = tf.layers.batch_normalization(inputs=conv3, training=training)

shape = tf.shape(conv3)

alpha = tf.Variable(10.)

y = tf.exp(conv3 / alpha)

y_sum = tf.reshape(tf.reduce_sum(y, axis=[1, 2]), [-1, 16])
denominator = tf.reshape(tf.tile(y_sum, [1, shape[1] * shape[2]]), shape)

y = tf.div(y, denominator)

i = tf.cast(tf.expand_dims(tf.reshape(tf.tile(tf.expand_dims(tf.tile(tf.range(0, 109, 1), [109]), -1), [1,16]), [109, 109, 16]), 0), tf.float32)
j = tf.transpose(i, [0, 2, 1, 3])

yi = tf.reduce_sum(tf.reshape(y * i, [-1, 109*109, 16]), [1])
yj = tf.reduce_sum(tf.reshape(y * j, [-1, 109*109, 16]), [1])

features = tf.concat([yi, yj], axis=1)

dense = tf.layers.dense(features,
                        3600,
                        use_bias=True,
                        kernel_initializer=initializer,
                        name="dense")

out = tf.reshape(dense, [-1, 60, 60, 1])
tf.summary.image("out", out, 1)

loss = tf.reduce_mean(tf.squared_difference(out, label))
tf.summary.scalar('loss', loss)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train = optimizer.minimize(loss)

sess = tf.Session()

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('log/train', sess.graph)

init = tf.global_variables_initializer()
sess.run(init)


big_images = [cv2.imread(file, 1) for file in glob.glob('./dataset/*/*.jpg')]
images = []

for image in big_images:
    images.append(cv2.resize(image, (240, 240)))


num_batch = int(len(images)/batch)
lab = np.zeros((batch, 60, 60, 1))
update = 0
for e in range(epochs):
    print(e)
    for index in range(num_batch):

        if index % 50 == 0:
            print(index)

        im = images[index:index+batch]
        for ii in range(batch):
            im_res = cv2.resize(images[index+ii], (60, 60))
            lab[ii] = np.expand_dims((im_res[:,:,0]+im_res[:,:,1]+im_res[:,:,2])/3., -1)

        _, summary = sess.run((train, merged), feed_dict={x: im, label: lab, training: True})
        train_writer.add_summary(summary, update)
        update += 1


reconstructed = sess.run((out), feed_dict={x: images[0:2], training: False})
plt.imshow(reconstructed[0,:,:,0])
plt.show()