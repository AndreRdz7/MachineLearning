# -*- coding: utf-8 -*-
"""
Handwritting recognition (numbers only)

@author: David André Rodríguez Méndez (AndreRdz7)
"""
# Import libraries
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from skimage import io
# Get dataset
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
print(len(mnist.train.images))
print(len(mnist.test.images))
im_temp = mnist.train.images[0]
io.imshow(np.reshape(im_temp,(28,28)))
# NN
dim_input = 784
n_categories = 10
x = tf.placeholder(tf.float32, [None, dim_input])
W = tf.Variable(tf.zeros([dim_input,n_categories]))
b = tf.Variable(tf.zeros([n_categories]))
softmax_args = tf.matmul(x,W) + b
y_hat = tf.nn.softmax(softmax_args)
# NN Training
y_ = tf.placeholder(tf.float32,[None,10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_hat), reduction_indices=[1]))
# tf.nn.softmax_cross_entropy_with_logits(softmax_args,y_)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
session = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(50000):
    batch_x, batch_y = mnist.train.next_batch(100)
    session.run(train_step, feed_dict={x:batch_x, y_: batch_y})
# NN Evaluation
correct_predictions = tf.equal(tf.argmax(y_hat,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions,tf.float32))
print(session.run(accuracy,feed_dict={x:mnist.test.images, y_:mnist.test.labels})*100)