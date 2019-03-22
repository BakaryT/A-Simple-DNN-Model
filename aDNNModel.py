#By Bakary TRAORE
#Project: https://github.com/aymericdamien/TensorFlow-Examples/
#Modified version of:
#A linear regression learning algorithm example using TensorFlow library.
#Author: Aymeric Damien
#Project: https://github.com/aymericdamien/TensorFlow-Examples/

from __future__ import print_function

import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random

# Parameters
learning_rate = 0.1
training_epochs = 1000
display_step = 50

# Training Data
train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")






#weights
W1=tf.Variable(rng.randn(),tf.float32)
W2=tf.Variable(rng.randn(),tf.float32)
W3=tf.Variable(rng.randn(),tf.float32)
Wo=tf.Variable(rng.randn(),tf.float32)
# biases
b1=tf.Variable(rng.randn(),tf.float32)
b2=tf.Variable(rng.randn(),tf.float32)
b3=tf.Variable(rng.randn(),tf.float32)
bo=tf.Variable(rng.randn(),tf.float32)

#Building the model

def model(data):
   
    # Hidden Layer 1
    l1 = tf.add(tf.multiply(data,W1), b1)
    l1 = tf.nn.tanh(l1)

    # Hidden Layer 2
    l2 = tf.add(tf.multiply(l1,W2), b2)
    l2 = tf.nn.tanh(l2)

    # Hidden Layer 3
    l3 = tf.add(tf.multiply(l2,W3), b3)
    l3 = tf.nn.tanh(l3)
    
    # Output Layer
    output = tf.multiply(l3,Wo) + bo

    return output

# Using the model

pred =model(X)

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# Gradient descent
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W1=", sess.run(W1), "b1=", sess.run(b1),"W2=", sess.run(W2), \
                "b2=", sess.run(b2),"W3=", sess.run(W3), "b3=", sess.run(b3),"Wo=", sess.run(Wo), "bo=", sess.run(bo))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W1=", sess.run(W1), "b1=", sess.run(b1),\
        "W2=", sess.run(W2), "b2=", sess.run(b2),"W3=", sess.run(W3), "b3=", sess.run(b3),"Wo=", sess.run(Wo), "bo=", sess.run(bo), '\n')

    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    
    # Creating the linear training function
    train_xx = W1 * train_X + b1
    train_xx = W2 * train_xx + b2
    train_xx = W3 * train_xx + b3
    train_xx = Wo * train_xx + bo
    plt.plot(train_X, sess.run(train_xx) , label='Fitted line')
    plt.legend()
    plt.show()

    # Testing example, as requested (Issue #2)
    test_X = numpy.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_Y = numpy.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))

    plt.plot(test_X, test_Y, 'bo', label='Testing data')

    # Creating the linear testing function
    train_xx = W1 * train_X + b1
    train_xx = W2 * train_xx + b2
    train_xx = W3 * train_xx + b3
    train_xx = Wo * train_xx + bo

    plt.plot(train_X, sess.run(train_xx), label='Fitted line')
    plt.legend()
    plt.show()
