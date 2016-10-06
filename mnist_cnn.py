# -- coding: utf-8 --
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

batch_size = 128
test_size = 256

def init_weights(name, shape):
    return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def model(X, w1, w2, w3, w4, w_o, p_keep_conv, p_keep_fc):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w1, strides=[1,1,1,1], padding='SAME'))        # (n,28,28,32)
    l1 = tf.nn.max_pool(l1a, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    # (n,14,14,32)
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, strides=[1,1,1,1], padding='SAME'))       # (n,14,14,64)
    l2 = tf.nn.max_pool(l2a, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    # (n,7,7,64)
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3, strides=[1,1,1,1], padding='SAME'))       # (n,7,7,128)
    l3 = tf.nn.max_pool(l3a, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    # (n,4,4,128)
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])                          # to (n,2048)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_fc)

    y = tf.matmul(l4, w_o)

    return y

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 28, 28, 1)             # 28x28x1 input image
teX = teX.reshape(-1, 28, 28, 1)

X = tf.placeholder('float', [None,28,28,1])
Y = tf.placeholder('float', [None,10])

w1 = init_weights('weight1', [3,3,1,32])    # [filter-width,filter-height,input,output]
w2 = init_weights('weight2', [3,3,32,64])
w3 = init_weights('weight3', [3,3,64,128])
w4 = init_weights('weight4', [4*4*128,625])
w_o = init_weights('weight5', [625,10])

p_keep_conv = tf.placeholder('float')
p_keep_fc = tf.placeholder('float')

pY = model(X, w1, w2, w3, w4, w_o, p_keep_conv, p_keep_fc)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pY, Y))
train = tf.train.AdamOptimizer(0.001).minimize(cost)
predict = tf.argmax(pY, 1)

with tf.Session() as sess:
    tf.initialize_all_variables().run()

    avg_cost = 0.
    for epoch in range(100):
        for start,end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
            sess.run(train, feed_dict={X:trX[start:end], Y:trY[start:end], 
                                    p_keep_conv:0.8, p_keep_fc:0.5})
            avg_cost += sess.run(cost, feed_dict={X:trX[start:end], Y:trY[start:end],
                                             p_keep_conv:0.8, p_keep_fc:0.5})

        print 'Epoch:', '%03d' % (epoch+1), 'cost=', '{:.9f}'.format(avg_cost/(len(trX)/batch_size))

        test_indices = np.arange(len(teX)) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        print(epoch, np.mean(np.argmax(teY[test_indices], axis=1) ==
                         sess.run(predict, feed_dict={X: teX[test_indices], Y: teY[test_indices],
                                                         p_keep_conv: 1.0, p_keep_fc: 1.0})))
