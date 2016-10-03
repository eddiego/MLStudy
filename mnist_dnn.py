from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

# hyper params
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# in/out
x = tf.placeholder('float', [None, 784])    # 28*28=784
y = tf.placeholder('float', [None, 10])     # 10 classes

# weights
w1 = tf.get_variable('weight1', shape=[784,256], initializer=tf.contrib.layers.xavier_initializer())
w2 = tf.get_variable('weight2', shape=[256,256], initializer=tf.contrib.layers.xavier_initializer())
w3 = tf.get_variable('weight3', shape=[256,256], initializer=tf.contrib.layers.xavier_initializer())
w4 = tf.get_variable('weight4', shape=[256,256], initializer=tf.contrib.layers.xavier_initializer())
w5 = tf.get_variable('weight5', shape=[256,256], initializer=tf.contrib.layers.xavier_initializer())
w6 = tf.get_variable('weight6', shape=[256,256], initializer=tf.contrib.layers.xavier_initializer())
w7 = tf.get_variable('weight7', shape=[256,10], initializer=tf.contrib.layers.xavier_initializer())

b1 = tf.Variable(tf.zeros([256]), name='bias1')
b2 = tf.Variable(tf.zeros([256]), name='bias1')
b3 = tf.Variable(tf.zeros([256]), name='bias1')
b4 = tf.Variable(tf.zeros([256]), name='bias1')
b5 = tf.Variable(tf.zeros([256]), name='bias1')
b6 = tf.Variable(tf.zeros([256]), name='bias1')
b7 = tf.Variable(tf.zeros([10]), name='bias1')

# model
dropout_rate = tf.placeholder('float')
L2 = tf.nn.dropout(tf.nn.relu(tf.matmul(x, w1) + b1), dropout_rate)
L3 = tf.nn.relu(tf.matmul(L2, w2) + b2)
L4 = tf.nn.dropout(tf.nn.relu(tf.matmul(L3, w3) + b3), dropout_rate)
L5 = tf.nn.relu(tf.matmul(L4, w4) + b4)
L6 = tf.nn.dropout(tf.nn.relu(tf.matmul(L5, w5) + b5), dropout_rate)
L7 = tf.nn.relu(tf.matmul(L6, w6) + b6)
hypothesis = tf.add(tf.matmul(L7, w7), b7)  # no need softmax along with 'with_logits'

# minimize cross-entropy
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# init. var
init = tf.initialize_all_variables()

# load data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# launch the graph
with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()

    print 'Start training'

    # training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)

        # loop over batch
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, dropout_rate:0.5})
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, dropout_rate:1}) / total_batch

        # display
        if epoch % display_step == 0:
            print 'Epoch:', '%04d' % (epoch+1), 'cost=', '{:.9f}'.format(avg_cost)

    print 'Optimization Finished'

    # get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print 'Label:', sess.run(tf.argmax(mnist.test.labels[r:r+1], 1))
    print 'Prediction:', sess.run(tf.argmax(hypothesis, 1), {x: mnist.test.images[r:r+1], dropout_rate:1})

    # show the image
    # plt.imshow(mnist.test.images[r:r+1].reshape(28,28), cmap='Greys', interpolation='nearest')
    # plt.show()

    # Test model
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print ("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels, dropout_rate:1}))

    saver.save(sess, 'mnist_dnn.ckpt')
