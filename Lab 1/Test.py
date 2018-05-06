from _ctypes import sizeof

import pandas as pd
from  RegressionTechniques import RegressionTechniques
from  CustomPlotter import CustomPlotter
import sys
import logging
import logging.config
import os
import json
import numpy as np

class Test(Exception):
    def readAndPrepCSVdata(self, target_field):
        try:
            """Read the data from the csv  file"""

            x = pd.read_csv ("parkinsons_updrs_test.csv")
        except IOError:

            sys.exit ( )

        try:

            # change the data type of the test_time float->int
            x["test_time"] = x["test_time"].astype ("int64")

            """Filter the data by subject and time of day."""
            df = x.groupby (["subject#", "test_time"]).mean ( ).reset_index ( )


            data_train = df.pipe (lambda d: d[d["subject#"] < 2])
            data_test = df.pipe (lambda d: d[d["subject#"] < 3])

            """Normalize data_train--> get mean values for measured features for a given subject and time"""
            """Normalisation is done by taking and applying the mean and standard deviation of the training data to 
            both training and testing data"""

            data_train_mean = data_train.mean ( )
            # the root square of the variance is the std_dev
            data_train_std_dev = data_train.std ( )
            data_train_norm = ((data_train - data_train_mean) / data_train_std_dev)
            # Normalize data_test with mean and variance of data_train
            data_test_norm = ((data_test - data_train_mean) / data_train_std_dev)

            """The chosen target can be either total UPDRS or Jitter % - Jitter % produced better results"""
            # ---Data to work with
            FO = target_field


            """Training Data is seperated into target and features"""
            self.y_train = data_train_norm[FO].values
            # get all features, by dropping the target column
            self.x_train = data_train_norm.drop ([FO], axis=1).values


            # Organise the test data too
            self.y_test = data_test_norm[FO].values
            self.x_test = data_test_norm.drop ([FO], axis=1).values

        except (ArithmeticError, OverflowError, FloatingPointError, ZeroDivisionError) as mathError:

            raise
        except Exception as e:

            raise


parkinson = Test ( )
parkinson.readAndPrepCSVdata ("total_UPDRS")



c = np.concatenate ((parkinson.x_train, parkinson.x_test), axis=0)

d= np.hsplit(c,np.array([-1]))

print(d[1])

