#lab4_arrythmia_nn_16_classes

# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

tf.set_random_seed(42)
np.random.seed(23)

#Read the file and put NaN instead of ?
xx=pd.read_csv('arrhythmia.data', sep=',', header=None, na_values=['?','\t?'])

x = xx.copy()
#Drop the rows with Nan values (axis=1 is columns):
x = x.dropna(axis=1)

#Remove the rows that are all 0 - b isa column
#  vector that sums on all rows
data = x.values
b = np.sum(np.abs(data), axis=0)
ii = np.where(b==0)
data = np.delete(data,ii,axis=1)

class_id = data[:,-1]
y = data[:,0:-1]

#data1 = data1-np.mean(data1,axis=0) #remove mean of feature values
#data2 = data1/np.sqrt(np.var(data1,axis=0)) #normalize the variance of each feature

Npatients, Nfeatures = y.shape
num_train = int(Npatients/2)

#Convert the vector Nx1 with the 16 classes given by the medical
#doctor into the matrix Nx16 with values 0 or 1
class_matrix = np.zeros((Npatients,16))
for i in range(Npatients):
    num = class_id[i]
    class_matrix[int(i)][int(num-1)] = 1
    
y_train = y[:num_train,:]
y_test = y[num_train:,:]
class_matrix_train = class_matrix[:num_train,:]
class_matrix_test = class_matrix[num_train:,:]

#%%
#--The neural network
#We put None because we run the neural network twice so we may have different no. patients
x = tf.placeholder(tf.float32, [None,Nfeatures]) #Nfeatures input nodes
t = tf.placeholder(tf.float32, [None,16]) #desired outputs - targets

#--neural network structure:
N_hidden_first_layer = Nfeatures//2


#%%
#x = tf.placeholder(tf.float32, [None, F]) # F input nodes
#t = tf.placeholder(tf.float32, [None, 16]) # 1 output node
w1 = tf.Variable(tf.truncated_normal(shape = [Nfeatures, N_hidden_first_layer], mean = 0.0, stddev = 1.0, dtype = tf.float32))
b1 = tf.Variable(tf.truncated_normal(shape = [1,N_hidden_first_layer], mean = 0.0, stddev = 1.0, dtype=tf.float32))
a1 = tf.matmul(x, w1) + b1
z1 = tf.nn.sigmoid(a1)

w2 = tf.Variable(tf.truncated_normal(shape = [N_hidden_first_layer, 16], mean = 0.0, stddev = 1.0, dtype = tf.float32))
b2 = tf.Variable(tf.truncated_normal(shape = [1,16], mean = 0.0, stddev = 1.0, dtype = tf.float32))
y = tf.nn.softmax(tf.matmul(z1, w2) + b2)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = t, logits = y))
optim = tf.train.GradientDescentOptimizer(0.6)
optim_op = optim.minimize(cost, var_list = [w1, b1, w2, b2])

# Running session
sess = tf.Session()
tf.global_variables_initializer().run(session = sess)
tf.local_variables_initializer().run(session = sess)
for i in range(300000):
    train_data = {x: y_train, t: class_matrix_train}
    sess.run([optim_op], feed_dict = train_data)
    if i%10000 == 0:
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
            
#%%
import seaborn as sn
plt.figure()
sn.set(font_scale=1.4)#for label size
sn.heatmap(conf_train, annot=True, annot_kws={"size": 12}, cmap="Greens", xticklabels=[i for i in range(1,17)], yticklabels=[i for i in range(1,17)])
plt.title("Confusion matrix Neural Network 16 classes-Training")
plt.xlabel("Class")
plt.ylabel("Class")


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
sn.set(font_scale=1.4)#for label size
sn.heatmap(conf_test, annot=True, annot_kws={"size": 12}, cmap="Greens", xticklabels=[i for i in range(1,17)], yticklabels=[i for i in range(1,17)])
plt.title("Confusion matrix Neural Network 16 classes-Testing")
plt.xlabel("Class")
plt.ylabel("Class")
plt.show()