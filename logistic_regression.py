import tensorflow as tf
import numpy as np

xy = np.loadtxt('train5.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1,len(x_data)],-1.0,1.0))

print x_data
print y_data

h = tf.matmul(W,X)
hypothesis = tf.div(1., 1.+tf.exp(-h))
cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))

optimizer = tf.train.AdamOptimizer(0.1)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print sess.run(W)

# train
for step in xrange(2001):
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if step % 20 == 0 :
        print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}),sess.run(W)

# test
print sess.run(hypothesis, feed_dict={X:[[1],[11],[7]]}) > 0.5
print sess.run(hypothesis, feed_dict={X:[[1],[1],[4]]}) > 0.5
print sess.run(hypothesis, feed_dict={X:[[1],[5],[2]]}) > 0.5

# accuracy
correct_prediction = tf.equal(tf.floor(hypothesis+0.5), Y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
print sess.run([hypothesis, tf.floor(hypothesis+0.5), correct_prediction, accuracy],
                feed_dict={X:x_data, Y:y_data})
print 'accuracy:', sess.run(accuracy, {X:x_data, Y:y_data})