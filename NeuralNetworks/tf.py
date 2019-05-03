# -*- coding: utf-8 -*-
"""
Neural Networks

@author: David André Rodríguez Méndez (AndreRdz7)
"""
# Import libraries
import tensorflow as tf
import numpy as np
import os
import random
import skimage.data as imd
from skimage import transform
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
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
# Neural Network
# Load images
def load_ml_data(data_directory):
    dirs = [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory,d))]
    labels = []
    images = []
    for d in dirs: 
        label_dir = os.path.join(data_directory,d)
        file_names = [os.path.join(label_dir,f) for f in os.listdir(label_dir) if f.endswith(".ppm")]
        for f in file_names:
            images.append(imd.imread(f))
            labels.append(int(d))
    return images,labels
main_dir = "./"
train_data_dir = os.path.join(main_dir,"Training")
test_data_dir = os.path.join(main_dir,"Testing")
images, labels = load_ml_data(train_data_dir)
print("Training images: ",len(images))
images = np.array(images)
labels = np.array(labels)
plt.hist(labels,len(set(labels)))
plt.show()
rand_signs = random.sample(range(0,len(labels)),6)
print(rand_signs)
for i in range(len(rand_signs)):
    temp_im = images[rand_signs[i]]
    plt.subplot(1,6,i+1)
    plt.axis("off")
    plt.imshow(images[rand_signs[i]])
    plt.subplots_adjust(wspace=0.5)
    plt.show()
    print("Froma:{0}, min:{1}, max:{2}".format(temp_im.shape,temp_im.min(),temp_im.max()))
unique_labels = set(labels)
plt.figure(figsize=(16,16))
i = 1
for label in unique_labels:
    temp_im = images[list(labels).index(label)]
    plt.subplot(8,8,i)
    plt.axis("off")
    plt.title("Clase:{0} ({1})".format(label,list(labels).count(label)))
    plt.imshow(temp_im)
plt.show()
# Image processing
w = 9999
h = 9999
for image in images:
    if image.shape[0] < h:
        h = image.shape[0]
    if image.shape[1] < w:
        w = image.shape[1]
print("Tamaño mínimo: {0}x{1}".format(h,w))
images30 = [transform.resize(image, (30,30)) for image in images]
images30 = np.array(images30)
images30 = rgb2gray(images30)
# Creating the model
x = tf.placeholder(dtype=tf.float32, shape=[None,30,30])
y = tf.placeholder(dtype=tf.int32,shape=[None])
images_flat = tf.contrib.layers.flatten(x)
logits = tf.contrib.layers.fully_connected(images_flat,62,tf.nn.relu)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits))
train_opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
final_pred = tf.argmax(logits,1)
accuracy = tf.reduce_mean(tf.cast(final_pred,tf.float32))
tf.set_random_seed(1234)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(1500):
    print("EPOCH: ",i)
    _, accuracy_val = sess.run([train_opt,accuracy],feed_dict={x:images30,y:list(labels)})
    _, loss_val = sess.run([train_opt,loss],feed_dict={x:images30,y:list(labels)})
    if i%10 == 0:
        print("Accuracy: ",accuracy_val)
        print("Loss: ",loss_val)
    print("Fin del epoch: ",i)
# Evaluation
sample_idx = random.sample(range(len(images30)),16)
sample_images = [images30[i] for i in sample_idx]
sample_labels = [labels[i] for i in sample_idx]
prediction = sess.run([final_pred],feed_dict={x:sample_images})[0]
print(prediction)
plt.figure(figsize=(16,9))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    predi = prediction[i]
    plt.subplot(4,4,i+1)
    plt.axis("off")
    color = "green" if truth==predi else "red"
    plt.text(32,15, "Real:      {0}\nPredicción:{1}".format(truth,predi),fontsize=14,color=color)
    plt.imshow(sample_images[i],cmap="gray")
plt.show()
test_images, test_labels = load_ml_data(test_data_dir)
test_images30 = [transform.resize(im,(30,30)) for im in test_images]
test_images30 = rgb2gray(np.array(test_images30))
prediction = sess.run([final_pred],feed_dict={x:test_images30})[0]
match_count = sum([int(10==lp) for 10,lp in zip(test_labels,prediction)])
print("Match count: ",match_count)
acc = match_count/len(test_labels)*100
print("Accuracy: ",accuracy)