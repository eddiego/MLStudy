import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.arange(0.,10)
y_data = [19, 20, 20.5, 21.5, 22, 23, 23, 25.5, 24, 25]

W = tf.Variable([-10.])
b = tf.Variable([-10.])
y = W * x_data + b

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.02)
train = optimizer.minimize(loss)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)

f, axarr = plt.subplots(1,5, sharey=True)

# Fit the line.
for step in xrange(801):
    sess.run(train)
    if step % 40 == 0:
        t0 = sess.run(b)
        t1 = sess.run(W)
        print 'step:{:}'.format(step) +' theta0=' + np.array_str(t0)+' theta1='+np.array_str(t1)
        if step % 200 == 0:
            axarr[step/200].plot(x_data,y_data,'ro', x_data, t1*x_data+t0)
            axarr[step/200].text(1,16, np.array_str(t0,precision=2)+' '+np.array_str(t1,precision=2))

plt.axis([-1,11,15,26]);
plt.show()
