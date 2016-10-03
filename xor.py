import tensorflow as tf
import numpy as np

xy = np.loadtxt('train7.txt', unpack=True)
x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1], (4,1))
# x_data = [[0.,0.],[0.,1.],[1.,0.],[1.,1.]]
# y_data = [[0.],[1.],[1.],[0.]]

X = tf.placeholder(tf.float32, name='x-input')
Y = tf.placeholder(tf.float32, name='y-input')

# w1 = tf.Variable(tf.random_uniform([2,4], -1.0, 1.0), name='weight1')
# w2 = tf.Variable(tf.random_uniform([4,3], -1.0, 1.0), name='weight2')
# w3 = tf.Variable(tf.random_uniform([3,1], -1.0, 1.0), name='weight3')
w1 = tf.get_variable('weight1', shape=[2,4], initializer=tf.contrib.layers.xavier_initializer())
w2 = tf.get_variable('weight2', shape=[4,3], initializer=tf.contrib.layers.xavier_initializer())
w3 = tf.get_variable('weight3', shape=[3,1], initializer=tf.contrib.layers.xavier_initializer())

b1 = tf.Variable(tf.zeros([4]), name='bias1')
b2 = tf.Variable(tf.zeros([3]), name='bias2')
b3 = tf.Variable(tf.zeros([1]), name='bias3')

L2 = tf.sigmoid(tf.matmul(X, w1) + b1)
L3 = tf.sigmoid(tf.matmul(L2, w2) + b2)
hypothesis = tf.sigmoid(tf.matmul(L3, w3) + b3)
# hypothesis = tf.nn.softmax(tf.matmul(L2, w2) + b2)    <- N/A. need more than 2 output

with tf.name_scope('cost') as scope:
    # minimize cross-entropy <- N/A. need more than 2 output
    # cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis)))
    # cost for sigmoid
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1 - hypothesis))
    cost_sum = tf.scalar_summary('cost', cost)

with tf.name_scope('train') as scope:
    optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
    # optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

# summary
w1_hist = tf.histogram_summary('weight1', w1)
w2_hist = tf.histogram_summary('weight2', w2)
w3_hist = tf.histogram_summary('weight3', w3)
b1_hist = tf.histogram_summary('bias1', b1)
b2_hist = tf.histogram_summary('bias2', b2)
b3_hist = tf.histogram_summary('bias3', b3)
y_hist =  tf.histogram_summary('y', Y)

# init var
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    # merge summary
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter('./logs/xor_logs', sess.graph)

    # train
    for step in xrange(12001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            summary = sess.run(merged, feed_dict={X: x_data, Y: y_data})
            writer.add_summary(summary, step)
            print step, sess.run(cost, feed_dict={X: x_data, Y: y_data})
            # print '  ', sess.run(w1), sess.run(w2), sess.run(w3)

    # test
    print 'test:', sess.run(hypothesis, feed_dict={X: x_data, Y: y_data})

    # accuracy
    correct_prediction = tf.equal(tf.floor(hypothesis+0.5), Y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    print sess.run((correct_prediction, accuracy), feed_dict={X:x_data, Y:y_data})
    print 'accuracy:', accuracy.eval({X:x_data, Y:y_data})