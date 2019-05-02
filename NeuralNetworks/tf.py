# -*- coding: utf-8 -*-
"""
Neural Networks

@author: David André Rodríguez Méndez (AndreRdz7)
"""
# Import libraries
import tensorflow as tf
# Create tensors
x1 = tf.constant([1,2,3,4,5])
x2 = tf.constant([6,7,8,9,10])
res = tf.multiply(x1,x2)
print(res)
# Create session
sess = tf.Session()
print(sess.run(res))
sess.close()
config = tf.ConfigProto(log_device_placement=True)
