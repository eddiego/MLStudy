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

# load MNIST data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# in/out
with tf.name_scope('input'):
    x = tf.placeholder('float', [None, 784])    # 28*28=784
    y = tf.placeholder('float', [None, 10])     # 10 classes

# weights
with tf.name_scope('weight'):
    w1 = tf.get_variable('weight1', shape=[784,256], initializer=tf.contrib.layers.xavier_initializer())
    w2 = tf.get_variable('weight2', shape=[256,256], initializer=tf.contrib.layers.xavier_initializer())
    w3 = tf.get_variable('weight3', shape=[256,10], initializer=tf.contrib.layers.xavier_initializer())

# bias
with tf.name_scope('bias'):
    b1 = tf.Variable(tf.zeros([256]), name='bias1')
    b2 = tf.Variable(tf.zeros([256]), name='bias2')
    b3 = tf.Variable(tf.zeros([10]), name='bias3')

# model
with tf.name_scope('model'):
    dropout_rate = tf.placeholder('float')
    L2 = tf.nn.dropout(tf.nn.relu(tf.matmul(x, w1) + b1), dropout_rate)
    L3 = tf.nn.dropout(tf.nn.relu(tf.matmul(L2, w2) + b2), dropout_rate) 
    hypothesis = tf.add(tf.matmul(L3, w3), b3)  # no need softmax along with 'with_logits'

# cost: minimize cross-entropy
with tf.name_scope('cost') as scope:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, y))

# train: optimizer
with tf.name_scope('train') as scope:
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# accuracy
with tf.name_scope('accuracy') as scope:
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# create summary for logging
tf.scalar_summary('cost', cost)
tf.scalar_summary('accuracy', accuracy)
tf.histogram_summary('y', y)

# merge all summary operator into single operator
summary_op = tf.merge_all_summaries()

# init. var operator
init = tf.initialize_all_variables()


# launch the graph
with tf.Session() as sess:
    # run init var op
    sess.run(init)

    # create saver
    saver = tf.train.Saver()

    # create log writer
    writer = tf.train.SummaryWriter('./summary/mnist_dnn', sess.graph)

    print 'Start training'

    # training cycle
    for epoch in range(training_epochs):

        # number of batches in one epoch
        batch_count = int(mnist.train.num_examples/batch_size)

        # loop over batch
        avg_cost = 0.
        for i in range(batch_count):
            # get a batch data
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            # do operator
            _, summary, bcost = sess.run([train_op, summary_op, cost], feed_dict={x: batch_xs, y: batch_ys, dropout_rate:0.5})
            
            # write log
            writer.add_summary(summary, training_epochs * batch_count + i)
            
            avg_cost += bcost / batch_count

        # display
        if epoch % display_step == 0:
            print 'Epoch:', '%03d' % (epoch+1), 'cost:', '{:.9f}'.format(avg_cost)

    print 'Optimization Finished'

    # get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print 'Label:', sess.run(tf.argmax(mnist.test.labels[r:r+1], 1))
    print 'Prediction:', sess.run(tf.argmax(hypothesis, 1), {x: mnist.test.images[r:r+1], dropout_rate:1})

    # show the image
    # plt.imshow(mnist.test.images[r:r+1].reshape(28,28), cmap='Greys', interpolation='nearest')
    # plt.show()

    # Test model
    print 'Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels, dropout_rate:1})

    saver.save(sess, 'ckpts/mnist_dnn.ckpt')
