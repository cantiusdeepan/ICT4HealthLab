import pandas as pd
from  RegressionTechniques import RegressionTechniques
from  CustomPlotter import CustomPlotter
import sys
import logging
import logging.config
import os
import json
import codecs

class Parkinsons_main(Exception):
    def __init__(self):
        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []
        self.logger = logging.getLogger(__name__)
        path = 'log_config.json'
        if os.path.exists(path):
            with open(path) as f:
                config = json.load(open(path))
            logging.config.dictConfig(config)

        else:
            logging.config.dictConfig({
                'version': 1,
                'disable_existing_loggers': False,  # this fixes the problem
                'formatters': {
                    'standard': {
                        'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
                    },
                },
                'handlers': {
                    'default': {
                        'level': 'INFO',
                        'class': 'logging.StreamHandler',
                    },
                },
                'loggers': {
                    '': {
                        'handlers': ['default'],
                        'level': 'INFO',
                        'propagate': True
                    }
                }
            })

    def readAndPrepCSVdata(self,target_field):

        try:
            """Read the data from the csv  file"""

            x = pd.read_csv("parkinsons_updrs.csv")
        except IOError:
            self.logger.error('Failed to read file', exc_info=True)
            sys.exit()

        try:

            # change the data type of the test_time float->int
            x["test_time"] = x["test_time"].astype("int64")

            """Filter the data by subject and time of day."""
            df = x.groupby(["subject#", "test_time"]).mean().reset_index()

            """Seperate part of data for training and another for testing"""
            self.logger.debug("Seperating training and testing data...")
            data_train = df.pipe(lambda d: d[d["subject#"] < 37])
            data_test = df.pipe(lambda d: d[d["subject#"] > 36])

            """Normalize data_train--> get mean values for measured features for a given subject and time"""
            """Normalisation is done by taking and applying the mean and standard deviation of the training data to 
            both training and testing data"""
            self.logger.debug("Normalising the features data set...")
            data_train_mean = data_train.mean()
            # the root square of the variance is the std_dev
            data_train_std_dev = data_train.std()
            data_train_norm = ((data_train - data_train_mean) / data_train_std_dev)
            # Normalize data_test with mean and variance of data_train
            data_test_norm = ((data_test - data_train_mean) / data_train_std_dev)

            """The chosen target can be either total UPDRS or Jitter % - Jitter % produced better results"""
            # ---Data to work with
            FO = target_field

            self.logger.debug("Removing the target field from training data...")
            """Training Data is seperated into target and features"""
            self.y_train = data_train_norm[FO].values
            # get all features, by dropping the target column
            self.x_train = data_train_norm.drop([FO], axis=1).values

            """Testing  Data is seperated into target and features"""
            self.logger.debug("Removing the target field from testing data...")
            # Organise the test data too
            self.y_test = data_test_norm[FO].values
            self.x_test = data_test_norm.drop([FO], axis=1).values
        except (ArithmeticError, OverflowError,FloatingPointError,ZeroDivisionError) as mathError:
            self.logger.error('Math error - Failed to prepare the csv data', exc_info=True)
            raise
        except Exception as e:
            self.logger.error('Failed to prepare the csv data', exc_info=True)
            raise





if __name__ == "__main__":
        #FO = "Jitter(%)"
        # FO = "total_UPDRS"
        target_field = "total_UPDRS"
        parkinson = Parkinsons_main()

        try:
            """Call the main csv reader function to read the data from the csv and prepare it for further use"""
            parkinson.logger.info("Starting to read the CSV....")
            parkinson.readAndPrepCSVdata(target_field)

            """Create objects for custom plotter class and regression techniques classes"""
            selfPlotter = CustomPlotter()
            parkinson.logger.info("Creating the regrssion and plotter objects....")
            mse = RegressionTechniques(parkinson.x_train,parkinson.x_test,parkinson.y_train,parkinson.y_test)
            gd = RegressionTechniques(parkinson.x_train, parkinson.x_test, parkinson.y_train, parkinson.y_test)
            sd = RegressionTechniques(parkinson.x_train, parkinson.x_test, parkinson.y_train, parkinson.y_test)

            """Calculate MSE with mean error taken per column"""
            parkinson.logger.info("Calculating MSE....")
            mse.calculate_MSE(mean_axis=0)

            """Plot actual training target vs estimated training target"""
            # --yhat_train versus y_train
            parkinson.logger.info("Plotting MSE....")
            selfPlotter.multiplot_sameX('MSE-yhat_train vs y_train','features',target_field
                                        ,originalData=parkinson.y_train,estimatedData=mse.yhat_train)

            """Plot actual testing target vs estimated testing target"""
            # --yhat_test versus y_test
            selfPlotter.multiplot_sameX('MSE-yhat_test vs y_test', 'features',target_field,
                                        originalData=parkinson.y_test, estimatedData=mse.yhat_test)

            """Plot the mean square error distribution for the training and testing data as histograms"""
            # the histogram of the error
            #multiplot_subHists(self, sp_rows, sp_cols, xlabel, ylabel, valuesPerSubplot, *bins, **yaxisTitleAndValue)
            selfPlotter.multiplot_subHists(1,2,'MSE-error values','MSE-error count',1, 50,50,
                                           training =mse.error_train,testing=mse.error_test)


            """Find the local optimum weights using gradient descent"""
            """Gamma value- the learning rate for penalizing errors - Value arrived at by testing different values"""
            gamma_value = 10 ** (-4)
            parkinson.logger.info("Calculating Gradient Descent....")
            """Stop error value- value identified from plotting the cost function value against no of iterations"""
            gd.gradientDescent(stop_error_value = 0.096,loop_limit=100000,gamma=gamma_value)

            """Plot the progression of error values for the gradient descent algorithm"""
            parkinson.logger.info("Plotting Gradient Descent....")
            selfPlotter.multiplot_sameX('Error_progression for Gradient descent','No.of Iterations','Change in E[w]',ChangeInError=gd.error)

            """Find the local optimum weights using Steepest descent(Newton's method)- Faster than gradient descent"""
            """Uses Hessian matrix(double derivative over the error in weights of the previous estimate) to reach the minimum soon"""
            parkinson.logger.info ("Calculating using Steepest Descent(Newton's method)....")
            """Stop error value- value identified from plotting the cost function value against no of iterations"""
            sd.steepestDescent_Hessian (stop_error_value=0.096, loop_limit=100000)

            """Plot the progression of error values for the gradient descent algorithm"""
            parkinson.logger.info ("Plotting Steepest Descent....")
            selfPlotter.multiplot_sameX ('Error_progression for Steepest descent', 'No.of Iterations', 'Change in E[w]',
                                         ChangeInError=sd.error)


        except (ArithmeticError, OverflowError,FloatingPointError,ZeroDivisionError) as mathError:
            parkinson.logger.error('Mathemtical Failure at main', exc_info=True)
            sys.exit()
        except (SystemExit, KeyboardInterrupt):
            sys.exit()
        except Exception as e:
            parkinson.logger.error('Common failure at main', exc_info=True)
            sys.exit()
        sys.exit (0)

