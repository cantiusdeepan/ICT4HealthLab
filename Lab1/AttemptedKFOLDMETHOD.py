import numpy as np
import logging
import os
from sklearn.model_selection import KFold
def kFoldDataSplitting_LastColumnY(self, data_to_be_split, fold_splits):
    kf = KFold (n_splits=fold_splits)
    k = 0
    x_train_data = np.zeros (1)
    x_test_data = np.zeros (1)
    y_train_data = np.zeros (1)
    y_test_data = np.zeros (1)

    for train, test in kf.split (data_to_be_split):

        train_data = np.array (self.data)[train]
        test_data = np.array (self.data)[test]
        """After combining test and train data-split target and domain based on k-fold split"""
        train_data_split = np.hsplit (train_data, np.array ([-1]))
        test_data_split = np.hsplit (test_data, np.array ([-1]))
        if (k == 0):
            np.put (x_train_data, k, np.array (train_data_split[0]))
            np.put (x_test_data, k, np.array (test_data_split[0]))
            np.put (y_train_data, k, np.array (train_data_split[1]))
            np.put (y_test_data, k, np.array (test_data_split[1]))
        else:
            x_train_data = np.append (x_train_data, np.array (train_data_split[0]))
            x_test_data = np.append (x_test_data, np.array (test_data_split[0]))
            y_train_data = np.append (y_train_data, np.array (train_data_split[1]))
            y_test_data = np.append (y_test_data, np.array (test_data_split[1]))

        k = k + 1
    return x_train_data, x_test_data, y_train_data, y_test_data