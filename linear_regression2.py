import tensorflow as tf
import numpy as np

from_file = True
if from_file:
    xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
    x_data = xy[0:-1]
    y_data = xy[-1]
    print x_data
    print y_data
else:
    x_data = [[1,1,1,1,1],  # for b
            [1.,0.,3.,0.,5.],[0.,2.,0.,4.,0.]]
    y_data = [1,2,3,4,5]

W = tf.Variable(tf.random_uniform([1,3],-1.0,1.0))
#b = tf.Variable(tf.random_uniform([1],-1.0,1.0))

hypothesis = tf.matmul(W, x_data) #+ b
cost = tf.reduce_mean(tf.square(hypothesis-y_data))
optimizer = tf.train.AdamOptimizer(0.01)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in xrange(2001):
    sess.run(train)
    if step % 20 == 0:
        print step, sess.run(cost), sess.run(W)#, sess.run(b)