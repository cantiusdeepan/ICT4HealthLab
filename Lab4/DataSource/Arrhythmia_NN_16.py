# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 19:34:20 2018

@author: Henrique
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

tf.set_random_seed(1234)
np.random.seed(567)

data_original = pd.read_csv("arrhythmia.data", header=None, na_values="?")

data = data_original.dropna(axis=1)
data = data.loc[:,(data != 0).any(axis=0)]

class_id = data[279]
y = data.loc[:,:278]
N, F = y.shape

class_matrix = np.zeros((N,16))
for i in range(N):
    num = class_id[i]
    class_matrix[i][num-1] = 1

nodes_hl = round(F/2)
num_train = int(N/2)

y_train = y.loc[:num_train-1,:]
y_test = y.loc[num_train:,:]
class_matrix_train = class_matrix[:num_train,:]
class_matrix_test = class_matrix[num_train:,:]

#%%
x = tf.placeholder(tf.float32, [None, F]) # F input nodes
t = tf.placeholder(tf.float32, [None, 16]) # 1 output node
w1 = tf.Variable(tf.truncated_normal(shape = [F, nodes_hl], mean = 0.0,
                                     stddev = 1.0, dtype = tf.float32))
b1 = tf.Variable(tf.truncated_normal(shape = [1,nodes_hl], mean = 0.0,
                                     stddev = 1.0, dtype=tf.float32))
a1 = tf.matmul(x, w1) + b1
z1 = tf.nn.sigmoid(a1)
w2 = tf.Variable(tf.truncated_normal(shape = [nodes_hl, 16], mean = 0.0,
                                     stddev = 1.0, dtype = tf.float32))
b2 = tf.Variable(tf.truncated_normal(shape = [1,16], mean = 0.0,
                                     stddev = 1.0, dtype = tf.float32))
y = tf.nn.softmax(tf.matmul(z1, w2) + b2)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels = t, logits = y))
optim = tf.train.GradientDescentOptimizer(0.6)
optim_op = optim.minimize(cost, var_list = [w1, b1, w2, b2])

# running the machine
sess = tf.Session()
tf.global_variables_initializer().run(session = sess)
tf.local_variables_initializer().run(session = sess)
for i in range(400000):
    train_data = {x: y_train, t: class_matrix_train}
    sess.run([optim_op], feed_dict = train_data)
    if i%1000 == 0:
        print (i, cost.eval(feed_dict = train_data, session = sess))
# calculating for training data
conf_matrix = tf.contrib.metrics.confusion_matrix(labels=tf.argmax(t, axis=1), predictions=tf.argmax(tf.round(y), axis=1), num_classes=16)
cm_train = conf_matrix.eval(feed_dict=train_data, session=sess)
a=np.sum(cm_train,1)
conf_train = np.empty((16,16), dtype=float)
for i in range(len(a)):
    for j in range(len(a)):
        if not a[i] == 0:
            conf_train[i,j] = float(cm_train[i,j])/a[i]
        else:
            conf_train[i,j] = 0
plt.figure()
plt.imshow(conf_train)
plt.title("Confusion matrix - Multiclass - NN - Training")
plt.xlabel("Class")
plt.ylabel("Class")
plt.colorbar()
#%%
#calculating for testing data
test_data = {x: y_test, t: class_matrix_test}
cm_test = conf_matrix.eval(feed_dict=test_data, session=sess)
a=np.sum(cm_test,1)
conf_test = np.empty((16,16), dtype=float)
for i in range(len(a)):
    for j in range(len(a)):
        if not a[i] == 0:
            conf_test[i,j] = float(cm_test[i,j])/a[i]
        else:
            conf_test[i,j] = 0
plt.figure()
plt.imshow(conf_test)
plt.title("Confusion matrix - Multiclass - NN - Testing")
plt.xlabel("Class")
plt.ylabel("Class")
plt.colorbar()
plt.show()