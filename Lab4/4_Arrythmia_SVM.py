import numpy as np
import pandas as pd
import seaborn as sn
from CustomPlotter import CustomPlotter1 as CustomPlotter
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle


# Read the cleaned and sorted Arrythmia file
data_df = pd.read_csv ('DataSource/arrhythmia_cleaned_sorted.csv', sep=',')
#Shuffling the rows in dataframe as we need to split training and test data
data_df = shuffle(data_df)
data2 = data_df.values
target = data2[:, -1]

# go from 16 classes to 2 classes. Previous class1 is class 1. Previous classes
# 2 to 16 is now Class 2
class_id = 1 * (target == 1) + 2 * (target > 1)
Y = data2[:, 0:-1]

Npatients, Nfeatures = Y.shape
train_seperator = int(Npatients//1.5)
print("train_seperator:",train_seperator)

x_train = Y[:train_seperator,:]
x_test = Y[train_seperator:,:]
y_train = class_id[:train_seperator]
y_test = class_id[train_seperator:]

print("Unique train classes:",np.unique(y_train))
print("Unique test classes:",np.unique(y_test))

#%% PCA
pca = PCA(n_components=60) #we are keeping 60 features
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

# %% SVM

parameters = {'kernel': ('linear', 'rbf'), 'C': [0.0001, 0.00015, 0.0002, 0.0005, 0.001, 0.01, 0.1, 1,2]}
svc = SVC ( )
clf = GridSearchCV (svc, parameters)
clf.fit (x_train, y_train)
est_class_id_train = clf.predict (x_train)
est_class_id_test = clf.predict (x_test)
print ("Score train:", clf.score (x_train, y_train))
print ("Score test:", clf.score (x_test, y_test))
print ('Best parameters:', clf.best_params_)

# %%
# Divide into train and test
true_positive_train = 0
true_positive_test = 0
false_positive_train = 0
false_positive_test = 0
true_negative_train = 0
true_negative_test = 0
false_negative_train = 0
false_negative_test = 0

# H1 - person is healthy, H2 - person is sick

# Training
for i in range (int (train_seperator)):
    if est_class_id_train[i] == 1:  # Tn
        if y_train[i] == 1:  # Dn
            true_negative_train += 1
        elif y_train[i] == 2:  # Dy
            false_negative_train += 1
    elif est_class_id_train[i] == 2:  # Tp
        if y_train[i] == 1:  # Dn
            false_positive_train += 1
        elif y_train[i] == 2:  # Dy
            true_positive_train += 1

sensitivity_train = float (true_positive_train) / (true_positive_train + false_negative_train)
specificity_train = float (true_negative_train) / (true_negative_train + false_positive_train)

print ("\nTrain_seperatoring set:")
print ("True positives:", true_positive_train)
print ("True negatives:", true_negative_train)
print ("False positives:", false_positive_train)
print ("False negatives:", false_negative_train)
print ("Sensitivity:", sensitivity_train)
print ("Specificity:", specificity_train)

# Testing
for i in range (int (Npatients - train_seperator)):
    if est_class_id_test[i] == 1:  # Tn
        if y_test[i] == 1:  # Dn
            true_negative_test += 1
        elif y_test[i] == 2:  # Dy
            false_negative_test += 1
    elif est_class_id_test[i] == 2:  # Tp
        if y_test[i] == 1:  # Dn
            false_positive_test += 1
        elif y_test[i] == 2:  # Dy
            true_positive_test += 1

sensitivity_test = float (true_positive_test) / (true_positive_test + false_negative_test)
specificity_test = float (true_negative_test) / (true_negative_test + false_positive_test)

print ("\nTesting set:")
print ("True positive:", true_positive_test)
print ("True negative:", true_negative_test)
print ("False positive:", false_positive_test)
print ("False negative:", false_negative_test)
print ("Sensitivity:", sensitivity_test)
print ("Specificity:", specificity_test)




