# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.utils import shuffle

tf.set_random_seed (97)
np.random.seed (45)

# Read the cleaned and sorted Arrythmia file
data_df = pd.read_csv ('DataSource/arrhythmia_cleaned_sorted.csv', sep=',')
# Shuffling the rows in dataframe as we need to split training and test data
data_df = shuffle (data_df)
data = data_df.values
target = data[:, -1]

# go from 16 classes to 2 classes. Previous class1 is class 1. Previous classes
# 2 to 16 is now Class 2
class_id = 0 * (target == 1) + 1 * (target > 1)
data1 = data[:, 0:-1]

data1 = data1 - np.mean (data1, axis=0)  # remove mean of feature values
data2 = data1 / np.sqrt (np.var (data1, axis=0))  # normalize the variance of each feature
Nf = data2.shape[1]
data = data2[:, 0:Nf]
Npatients, Nfeatures = data.shape
N_train = int(Npatients // (5.0/3.0))  # patients for training
N_test = Npatients - N_train  # patients for testing
data_train = data[0:N_train, :]
data_test = data[N_train:Npatients, :]
class_id_train = class_id[0:N_train]
class_id_test = class_id[N_train:Npatients]

# %%
# --The neural network
# We put None because we run the neural network twice so we may have different no. patients
x = tf.placeholder (tf.float32, [None, Nfeatures])  # inputs
t = tf.placeholder (tf.float32, [None, 1])  # desired outputs - targets

# --neural network structure:
N_hidden_first_layer = Nfeatures // 2
N_hidden_last_layer = 1

# initialize vector of weights [1,3] of random numbers according to Gaussian distr
w1 = tf.Variable (tf.random_normal (shape=[Nfeatures, N_hidden_first_layer], mean=0.0, stddev=1.0, dtype=tf.float32))
# biases vector also random
b1 = tf.Variable (tf.random_normal (shape=[N_hidden_first_layer, ], mean=0.0, stddev=1.0, dtype=tf.float32))
# you have to work with multiplication for tensorflow, they are tensors and not ndarrays!
a1 = tf.matmul (x, w1) + b1
# you have the function sigmoid inside Tensorflow nn.sigmoid
z1 = tf.nn.sigmoid (a1)

# %%
w2 = tf.Variable (tf.random_normal ([N_hidden_first_layer, 1], mean=0.0, stddev=1.0, dtype=tf.float32))
b2 = tf.Variable (tf.random_normal ([1, ], mean=0.0, stddev=1.0, dtype=tf.float32))
# neural network output
y = tf.nn.sigmoid (tf.matmul (z1, w2) + b2)

# -- optimizer structure
# Objective function:
cost = tf.reduce_mean (tf.nn.sigmoid_cross_entropy_with_logits (labels=t, logits=y))
# use Gradient Descent in the training phase
optim = tf.train.GradientDescentOptimizer (0.6)
# minimize the objective func changing w1,b1,w2,b2
optim_op = optim.minimize (cost, var_list=[w1, b1, w2, b2])

# %%

# definition of true and false positives and negatives
true_neg, true_neg_op = tf.metrics.true_negatives (labels=t, predictions=tf.round (y))
true_pos, true_pos_op = tf.metrics.true_positives (labels=t, predictions=tf.round (y))
false_neg, false_neg_op = tf.metrics.false_negatives (labels=t, predictions=tf.round (y))
false_pos, false_pos_op = tf.metrics.false_positives (labels=t, predictions=tf.round (y))

# %%
# --Initialize session
sess = tf.Session ( )

# --initialize
tf.global_variables_initializer ( ).run (session=sess)
tf.local_variables_initializer ( ).run (session=sess)

# -- run learning machine
for i in range (40000):
    xval = data_train  # (226,257)
    # xval = np.reshape(xval, (Npatients,1))
    tval = class_id_train.reshape (N_train, 1)  # (226,1)
    # train
    train_data = {x: xval, t: tval}
    sess.run ([optim_op], feed_dict=train_data)

    # print cost func every 100 steps to check if gradient descent is good and cost func is decreasing
    if i % 10000 == 0:
        print (i, cost.eval (feed_dict=train_data, session=sess))

# print(sess.run(w1),sess.run(b1))
# print(sess.run(w2),sess.run(b2))


# %%
# calculating for training data

tn_train = sess.run (true_neg_op, feed_dict=train_data)
tp_train = sess.run (true_pos_op, feed_dict=train_data)
fn_train = sess.run (false_neg_op, feed_dict=train_data)
fp_train = sess.run (false_pos_op, feed_dict=train_data)
train_0_count = (class_id_train == 0).sum ( )
train_1_count = (class_id_train == 1).sum ( )
true_neg_train = tn_train / train_0_count
true_pos_train = tp_train / train_1_count
false_neg_train = fn_train / train_1_count
false_pos_train = fp_train / train_0_count
# print('\nNeural Networks:\n')
print ('\nTraining phase:')
print ('True positive probability: ', true_pos_train)
print ('True negative probability: ', true_neg_train)
print ('False positive probability: ', false_pos_train)
print ('False negative probability: ', false_neg_train)
sensitivity_train = float (true_pos_train) / (true_pos_train + false_neg_train)
specificity_train = float (true_neg_train) / (true_neg_train + false_pos_train)
print ('Training sensitivity: ', sensitivity_train)
print ('Training specificity: ', specificity_train)
# %%
# calculating for testing data
test_data = {x: data_test, t: class_id_test.reshape (N_test, 1)}
tf.local_variables_initializer ( ).run (session=sess)
tn_test = sess.run (true_neg_op, feed_dict=test_data)
tp_test = sess.run (true_pos_op, feed_dict=test_data)
fn_test = sess.run (false_neg_op, feed_dict=test_data)
fp_test = sess.run (false_pos_op, feed_dict=test_data)
true_neg_test = tn_test / (class_id_test == 0).sum ( )
true_pos_test = tp_test / (class_id_test == 1).sum ( )
false_neg_test = fn_test / (class_id_test == 1).sum ( )
false_pos_test = fp_test / (class_id_test == 0).sum ( )
print ('Testing phase:')
print ('True positive probability: ', true_pos_test)
print ('True negative probability: ', true_neg_test)
print ('False positive probability: ', false_pos_test)
print ('False negative probability: ', false_neg_test)
sensitivity_test = float (true_pos_test) / (true_pos_test + false_neg_test)
specificity_test = float (true_neg_test) / (true_neg_test + false_pos_test)
print ('Testing sensitivity: ', sensitivity_test)
print ('Testing specificity: ', specificity_test)


