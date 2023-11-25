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

#read data from file
data_input = load_data_mnist('https://github.com/egasgira/TransferLearning/raw/master/mnist.pkl.gz')
#FYI data = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
data = data_input[0]
#print ( N.shape(data[0][0])[0] )
#print ( N.shape(data[0][1])[0] )

#data layout changes since output should an array of 10 with probabilities
real_output = N.zeros( (N.shape(data[0][1])[0] , 10), dtype=N.float )
for i in range ( N.shape(data[0][1])[0] ):
  real_output[i][data[0][1][i]] = 1.0  

#data layout changes since output should an array of 10 with probabilities
real_check = N.zeros( (N.shape(data[2][1])[0] , 10), dtype=N.float )
for i in range ( N.shape(data[2][1])[0] ):
  real_check[i][data[2][1][i]] = 1.0


epochs_lim = 500
loss_list_all = []
acc_list_all = []
epoch = []
lr_list = [0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005]

for lr in lr_list:

  #set up the computation. Definition of the variables.
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.nn.softmax(tf.matmul(x, W) + b)
  y_ = tf.placeholder(tf.float32, [None, 10])

  cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

  # Gradient descent
  #name_opt = "Gradient Descent Optimizer"
  #train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)
 
  # Momentum
  #name_opt = 'Momentum'
  #train_step = tf.train.MomentumOptimizer(lr, momentum=0.9).minimize(cross_entropy)
  
  # Adam
  #name_opt = 'Adam'
  #train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

  # Nesterov Accelerated Gradient (NAG)
  #name_opt = 'Nesterov Accelerated (NAG)'
  #train_step = tf.train.MomentumOptimizer(lr, momentum=0.9,use_nesterov=True).minimize(cross_entropy)

  # ADAGRAD
  #name_opt = 'ADAGRAD'
  #train_step = tf.train.AdagradOptimizer(lr).minimize(cross_entropy)

  # Adadelta
  #name_opt = 'Adadelta'
  #train_step = tf.train.AdadeltaOptimizer(lr).minimize(cross_entropy)

  # Adam
  #name_opt = 'Adam'
  #train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

  # RMSP
  #name_opt = 'RMSP'
  #train_step = tf.train.RMSPropOptimizer(lr).minimize(cross_entropy)

  # Ftrl
  #name_opt = 'Ftrl'
  #train_step = tf.train.FtrlOptimizer(learning_rate=lr, l1_regularization_strength=0.001, l2_regularization_strength=0.001).minimize(cross_entropy)

  # ProximalAdagrad
  name_opt = 'ProximalAdagrad'
  train_step = tf.train.ProximalAdagradOptimizer(learning_rate=lr, l1_regularization_strength=0.0001).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  train_accs = []
  train_losses = []
  valid_accs = []
  valid_losses = []

  # Define the accuracy and cross entropy tensors
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


  #TRAINING PHASE
  print("TRAINING")

  for i in range(epochs_lim):
    batch_xs = data[0][0][100*i:100*i+100]
    batch_ys = real_output[100*i:100*i+100]
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Compute training accuracy and loss
    train_acc, train_loss = sess.run([accuracy, cross_entropy], feed_dict={x: data[0][0], y_: real_output})
    # Compute validation accuracy and loss
    valid_acc, valid_loss = sess.run([accuracy, cross_entropy], feed_dict={x: data[1][0], y_: real_check[:N.shape(data[1][1])[0]]})

    # Append current epoch's training and validation accuracy and loss to respective lists
    train_accs.append(train_acc)
    train_losses.append(train_loss)
    valid_accs.append(valid_acc)
    valid_losses.append(valid_loss)
    if lr == 0.01: epoch.append(i)


  #CHECKING THE ERROR
  print("ERROR CHECK")

  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: data[2][0], y_: real_check}))

  #CHECKING THE ERROR
  print("ERROR CHECK")
  print(f"Train accuracy: {train_acc}")
  print(f"Train loss: {train_loss}")
  print(f"Validation accuracy: {valid_acc}")
  print(f"Validation loss: {valid_loss}")

  loss_list_all.append(train_losses)
  acc_list_all.append(train_accs)

# Plot Train Loss
plt.figure()
for i, loss_list in enumerate(loss_list_all):
  plt.plot(epoch, loss_list, color="C"+str(i), linestyle="-", label="LR={}".format(lr_list[i]))
  plt.text(len(loss_list)-1, loss_list[-1], "----->Loss:  {:.2f}".format(loss_list[-1])+"LR: "+str(lr_list[i]))
plt.title(name_opt + " - Loss - Epochs="+str(epochs_lim))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='center right')
plt.savefig(name_opt + 'Train_Loss_'+str(epochs_lim)+'.png')

# Plot Val Loss
plt.figure()
for i, acc_list in enumerate(acc_list_all):
  plt.plot(epoch, acc_list, color="C"+str(i), linestyle="-", label="LR={}".format(lr_list[i]))
  plt.text(len(acc_list)-1, acc_list[-1], "----->Acc:  {:.2f}".format(acc_list[-1])+"-LR:"+str(lr_list[i]))
plt.title(name_opt + " - ACC - Epochs="+str(epochs_lim))
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='center right')
plt.savefig(name_opt + 'ACC_Loss_'+str(epochs_lim)+'.png')