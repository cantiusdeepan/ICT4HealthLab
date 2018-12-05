# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 15:16:21 2018

@author: Henrique
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_original = pd.read_csv("arrhythmia.data", header=None, na_values="?")

data = data_original.dropna(axis=1)
data = data.loc[:,(data != 0).any(axis=0)]
#data.loc[data[279]>1,279] = 2

class_id = data[279]
y = data.loc[:,:278]

y_i = []
x_i = []
for i in range(1, 17):
    y_i.append(y.loc[class_id==i])
    x_i.append(y_i[i-1].mean(axis=0))

#%% min distance
dif_i = []
for i in range(0, 16):
    dif_i.append(y.apply(lambda x: x-x_i[i], axis=1))

dist_i = []
for i in range(0, 16):
    dist_i.append(dif_i[i].apply(lambda x: x*x, axis=1).sum(axis=1))

est_class_id = []
for i in range(452):
    min_dist = 9999999
    class_est = -1
    for j in range(16):
        if dist_i[j][i] < min_dist and dist_i[j][i] != 0:
            min_dist = dist_i[j][i]
            class_est = j+1
    est_class_id.append(class_est)

#%% Confusion matrix
correct_pred = 0    

conf_matrix = np.zeros((16,16))
for k in range(452):
     i = class_id[k]
     j = est_class_id[k]
     conf_matrix[i-1][j-1] += 1
     if i == j:
         correct_pred += 1

for i in range(16):
    total = conf_matrix[i].sum()
    conf_matrix[i] = conf_matrix[i]/total
    
conf_matrix = np.nan_to_num(conf_matrix)

plt.figure()
plt.imshow(conf_matrix)
plt.colorbar()
plt.title("Confusion matrix - Multiclass - MD")
plt.xlabel("Class")
plt.ylabel("Class")
plt.show()

