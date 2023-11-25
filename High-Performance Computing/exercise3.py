import tarfile
import re
import copy
import pickle
import argparse
import os.path
import sys
import time
import gzip
import configparser
import csv
import numpy as np
import requests
import io

def load_data_mnist(dataset):
    response = requests.get(dataset)
    content = response.content
    f = gzip.GzipFile(fileobj=io.BytesIO(content), mode='rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    f.close()

    n_in = 28
    in_channel = 1

    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval, n_in, in_channel, 10


def load_data_cifar10(dataset):
    train_dic = dict()
    f = tarfile.open(dataset, 'r')
    for member in f.getmembers():
        match1 = re.search(r".*\/data_batch_.*", member.name)
        match2 = re.search(r".*\/test_batch", member.name)
        if match1 is not None:
            print("Training: extracting {} ...".format(member.name))
            ef = f.extractfile(member)
            train_tmp = pickle.load(ef, encoding='latin1')
            if bool(train_dic) is False:
                train_dic = train_tmp
            else:
                train_dic['data'] = np.append(train_dic['data'], train_tmp['data'], axis=0)
                train_dic['labels'] = np.append(train_dic['labels'], train_tmp['labels'], axis=0)
        elif match2 is not None:
            print("Testing/Validating: extracting {} ...".format(member.name))
            ef = f.extractfile(member)
            test_dic = pickle.load(ef, encoding='latin1')
            test_dic['labels'] = np.array(test_dic['labels'])
    f.close()

    n_in = 32
    in_channel = 3

    def shared_dataset(data_dic, borrow=True):
        data_x = data_dic['data'].astype(np.float32)
        data_x /= 255.0  # data_x.max()
        data_y = data_dic['labels'].astype(np.int32)
        return data_x, data_y

    train_set_x, train_set_y = shared_dataset(train_dic)
    test_set_x, test_set_y = shared_dataset(test_dic)
    # valid_set_x, valid_set_y = shared_dataset(valid_dic)

    rval = [(train_set_x, train_set_y), (test_set_x, test_set_y), (test_set_x, test_set_y)]
    return rval, n_in, in_channel, 10

import tensorflow as tf
if tf.__version__.startswith('2.'):
    tf = tf.compat.v1
tf.disable_v2_behavior()
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as N
import time
import random

#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#read data from file
data_input = load_data_mnist('https://github.com/egasgira/TransferLearning/raw/master/mnist.pkl.gz')
#FYI data = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
data = data_input[0]
#print ( N.shape(data[0][0])[0] )
#print ( N.shape(data[0][1])[0] )

N_GPU = 4

#data layout changes since output should an array of 10 with probabilities
real_output = N.zeros( (N.shape(data[0][1])[0] , 10), dtype=N.float )
for i in range ( N.shape(data[0][1])[0] ):
  real_output[i][data[0][1][i]] = 1.0  


#data layout changes since output should an array of 10 with probabilities
real_check = N.zeros( (N.shape(data[2][1])[0] , 10), dtype=N.float )
for i in range ( N.shape(data[2][1])[0] ):
  real_check[i][data[2][1][i]] = 1.0


#set up the computation. Definition of the variables.
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
y_ = tf.placeholder(tf.float32, [None, 10])



#declare weights and biases
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


#convolution and pooling
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')




#First convolutional layer: 32 features per each 5x5 patch
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])


#Reshape x to a 4d tensor, with the second and third dimensions corresponding to image width and height.
#28x28 = 784
#The final dimension corresponding to the number of color channels.
x_image = tf.reshape(x, [-1, 28, 28, 1])


#We convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool. 
#The max_pool_2x2 method will reduce the image size to 14x14.

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)



#Second convolutional layer: 64 features for each 5x5 patch.
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


#Densely connected layer: Processes the 64 7x7 images with 1024 neurons
#Reshape the tensor from the pooling layer into a batch of vectors, 
#multiply by a weight matrix, add a bias, and apply a ReLU.
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#drop_out
keep_prob = 0.5
keep_prob = tf.placeholder_with_default(keep_prob, shape=())

#keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


#Readout Layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#Per_image_crossentropy
cross_entropy_local = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)

#Crossentropy
cross_entropy = tf.reduce_mean(cross_entropy_local)
#    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
batch = tf.Variable(0)
train_size = N.shape(data[0][0])[0]
batch_size = 150
learning_rate = tf.train.exponential_decay(
  10e-4,                # Base learning rate.
  batch * batch_size,  # Current index into the dataset.
  train_size,          # Decay step.
  0.95,                # Decay rate.
  staircase=True)

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#/Users/joaovalerio/Downloads/TensorFlow-course-P9/multilayer.py


with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
    # TRAIN
    print("TRAINING")

    start_time = time.time()

    for i in range(5000):
        # Randomize the batch start and end indices
        batch_ini = random.randint(0, len(data[0][0]) - batch_size)
        batch_end = batch_ini + batch_size

        for j in range(N_GPU):
            with tf.device('/gpu:%d' % j):
                batch_xs = data[0][0][batch_ini:batch_end]
                batch_ys = real_output[batch_ini:batch_end]

        if i % 10 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch_xs, y_: batch_ys, keep_prob: 1.0})
            print('step %d, training accuracy %g Batch [%d,%d]' % (i, train_accuracy, batch_ini, batch_end))

        train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

    print("Training Time: %.3f seconds" % (time.time() - start_time))
    # TEST
    print("TESTING")

    train_accuracy = accuracy.eval(feed_dict={x: data[2][0], y_: real_check, keep_prob: 1.0})
    print('test accuracy %.3f' % (train_accuracy))