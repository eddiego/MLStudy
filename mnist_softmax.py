from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

# hyper params
learning_rate = 0.01
training_epochs = 5
batch_size = 100
display_step = 1

# in/out
x = tf.placeholder('float', [None, 784])    # 28*28=784
y = tf.placeholder('float', [None, 10])     # 10 classes

# weights
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# model
activation = tf.nn.softmax(tf.matmul(x, W) + b)     # softmax

# minimize cross-entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(activation), reduction_indices=1))
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(activation, y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

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
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys}) / total_batch

        # display
        if epoch % display_step == 0:
            print 'Epoch:', '%04d' % (epoch+1), 'cost=', '{:.9f}'.format(avg_cost)

    print 'Optimization Finished'

    # get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print 'Label:', sess.run(tf.argmax(mnist.test.labels[r:r+1], 1))
    print 'Prediction:', sess.run(tf.argmax(activation, 1), {x: mnist.test.images[r:r+1]})

    # show the image
    plt.imshow(mnist.test.images[r:r+1].reshape(28,28), cmap='Greys', interpolation='nearest')
    plt.show()

    # Test model
    correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print ("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

    saver.save(sess, 'train_test_model.ckpt')
