import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#Read the file and put NaN instead of ?
xx=pd.read_csv('arrhythmia.data', sep=',', header=None, na_values=['?','\t?'])

df = xx.copy()
#Drop the rows with Nan values (axis=1 is columns):
df = df.dropna(axis=1)

#Remove the rows that are all 0 - b is a column vector that sums on all rows
data = df.values
b = np.sum(np.abs(data), axis=0)
ii = np.where(b==0)
data = np.delete(data,ii,axis=1)

#Sort data
data_df = pd.DataFrame(data)
data_df = data_df.sort_values(by=[257])
#data2 = data_df.values

#Divide into target and features
class_id = data_df.iloc[:,-1]
class_id = class_id.values
Y = data_df.iloc[:, 0:-1]
Y = Y.values
bla_Y = Y.copy()
bla_class = class_id.copy()

#%% Henrique

y_i = []
x_i = []
for i in range(1,17):
    if i==11 or i==12 or i==13:
        y_i.append(np.zeros((1,257)))
    else:
        separator = int(np.where(bla_class==i)[0][-1])+1
        y_i.append(bla_Y[0:separator,:])
        bla_Y = bla_Y[separator:,:]
        bla_class = bla_class[separator:]
      
    x_i.append(np.mean(y_i[i-1],axis=0))

#%% min distance

diff_i = []
for i in range(0, len(x_i)):
    diff_i.append(np.subtract(Y,x_i[i]))

dist_i = []
for i in range(0, len(x_i)):
    dist_i.append(np.sum(np.square(diff_i[i]),axis=1))

est_class_id = []
for i in range(len(class_id)):
    min_dist = 9999999
    class_est = -1
    for j in range(len(x_i)):
        if dist_i[j][i] < min_dist and dist_i[j][i] != 0:
            min_dist = dist_i[j][i]
            class_est = j+1
    est_class_id.append(class_est)
    
est_class_id = np.asarray(est_class_id)

#%% Confusion matrix
#    In case of multiclass classification, the confusion matrix
#is measured: the element in position i; j of the matrix is the probability
#that the estimated class is j given that the true class is i (the sum of the
#elements in a row must be 1).
    
correct_pred = 0    

conf_matrix = np.zeros((16,16))
for k in range(class_id.shape[0]):
    i = class_id[k]
    j = est_class_id[k]
    conf_matrix[int(i)-1][int(j)-1] += 1
    if i == j:
        correct_pred += 1

for i in range(16):
    total = conf_matrix[i].sum()
    conf_matrix[i] = conf_matrix[i]/total
    
conf_matrix = np.nan_to_num(conf_matrix)

import seaborn as sn
plt.figure(figsize = (12,10))
sn.set(font_scale=1.4)#for label size
sn.heatmap(conf_matrix, annot=True, annot_kws={"size": 12}, cmap="Greens", xticklabels=[i for i in range(1,17)], yticklabels=[i for i in range(1,17)])# font size
plt.title("Confusion matrix of Minimum distance criteria with 16 classes")
plt.xlabel("Class")
plt.ylabel("Class")
plt.show()
plt.savefig('Min distance 16')


