import pandas as pd
from  RegressionTechniques import RegressionTechniques
from  CustomPlotter import CustomPlotter
import sys
import logging
import logging.config
import os
import json
import codecs
import numpy as np

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
            data = df.pipe (lambda d: d[d["subject#"] > 0])
            data_train = data_train.drop (['subject#', 'age', 'sex', 'test_time'], axis=1)
            data_test = data_test.drop (['subject#', 'age', 'sex', 'test_time'], axis=1)


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
            data_norm = ((data - data_train_mean) / data_train_std_dev)
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

            self.y_data = data_norm[FO].values
            self.x_data = data_norm.drop([FO], axis=1).values


            self.y_data = np.reshape(self.y_data,(int(self.y_data.shape[0]),1))

            self.data = np.concatenate ((self.x_data, self.y_data), axis=1)


        except (ArithmeticError, OverflowError,FloatingPointError,ZeroDivisionError) as mathError:
            self.logger.error('Math error - Failed to prepare the csv data', exc_info=True)
            raise
        except Exception as e:
            self.logger.error('Failed to prepare the csv data', exc_info=True)
            raise





if __name__ == "__main__":
        #FO = "Jitter(%)"
        #FO = "total_UPDRS"
        target_field = "Jitter(%)"
        target_field = "total_UPDRS"
        parkinson = Parkinsons_main()

        try:
            """Call the main csv reader function to read the data from the csv and prepare it for further use"""
            parkinson.logger.info("Starting to read the CSV....")
            parkinson.readAndPrepCSVdata(target_field)

            """Create objects for custom plotter class and regression techniques classes"""
            selfPlotter = CustomPlotter()
            parkinson.logger.info("Creating the regrssion and plotter objects....")

            mse = RegressionTechniques(parkinson.x_train, parkinson.x_test, parkinson.y_train, parkinson.y_test,parkinson.data)

            gd = RegressionTechniques(parkinson.x_train, parkinson.x_test, parkinson.y_train, parkinson.y_test,parkinson.data)

            sd = RegressionTechniques(parkinson.x_train, parkinson.x_test, parkinson.y_train, parkinson.y_test,parkinson.data)

            #rr =  RegressionTechniques(parkinson.x_train, parkinson.x_test, parkinson.y_train, parkinson.y_test,parkinson.data)
            # <editor-fold desc="Description">
            """########################################################################################################################################"""
            """MSE"""

            """Calculate MSE with mean error taken per column"""
            parkinson.logger.info("Calculating MSE....")
            mse.calculate_MSE(mean_axis=0)


            """Plot the mean square error distribution for the training and testing data as histograms"""
            """MSE- the histogram of the error- comparing ther error values for test and training data"""
            #multiplot_subHists(self, sp_rows, sp_cols, xlabel, ylabel, valuesPerSubplot, *bins, **yaxisTitleAndValue)
            selfPlotter.multiplot_subHists('MSE- Training error vs Testing error',1,2,'MSE-error values','MSE-error count',1, 50,50,
                                           training =mse.error_train_diff,testing=mse.error_test_diff)
            #Plot and Compare estimated values with regression values
            selfPlotter.graph_plot (mse.yhat_train, mse.y_train, "Actual Values", "Estimated Values",
                                    "MSE - Y_Train vs Yhat_Train")
            selfPlotter.graph_plot ( mse.yhat_test,mse.y_test, "Actual Values","Estimated Values" , "MSE - Y_Test vs Yhat_Test ")

            selfPlotter.weight_graph_plot(mse.w, "Features", "Weight of Features", "MSE- Weights of Features", weight=1)
            """########################################################################################################################################"""
            """Gradient Descent"""

            """Find the local optimum weights using gradient descent"""
            """Gamma value- the learning rate for penalizing errors - Value arrived at by testing different values"""
            gamma_value = 10 ** (-4)
            parkinson.logger.info("Calculating Gradient Descent....")
            """Stop error value- value identified from plotting the cost function value against no of iterations"""
            gd.gradientDescent(stopping_condn = 10**-8,loop_limit=10**6,gamma=gamma_value)

            """Plot the progression of error values for the gradient descent algorithm"""
            parkinson.logger.info("Plotting Gradient Descent....")
            selfPlotter.multiplot_sameX('MSE_progression for Gradient descent','No.of Iterations','Change in E[w]',ChangeInError_Training=gd.error_train,ChangeInError_Testing=gd.error_test)
            #selfPlotter.multiplot_sameX('Parameter change for Gradient descent','No.of Iterations','Change in W',ChangeInW=gd.w_vector)
            ##def multiplot_subplots_withXaxis(self, title,sp_rows, sp_cols, xlabel, ylabel, valuesPerSubplot, *xAxisValue,##**yaxisTitleAndValue)
            # selfPlotter.multiplot_subplots_withXaxis( 'Gradient Descent - W vs Error',1, 1, 'W', 'E[W]', 1,gd.w_vector,
            #                                  gradient_function = gd.cost_function_gradient_vector)

            """Gradient DescEnt- the histogram of the error- comparing ther error values for test and training data"""
            #multiplot_subHists(self, sp_rows, sp_cols, xlabel, ylabel, valuesPerSubplot, *bins, **yaxisTitleAndValue)
            selfPlotter.multiplot_subHists('Gradient descent- Training error vs Testing error',1,2,'GD-error values','GD-error count',1, 50,50,
                                           training =gd.error_train_diff,testing=gd.error_test_diff)

            selfPlotter.graph_plot (gd.yhat_train, gd.y_train, "Actual Values", "Estimated Values",
                                    "GD - Y_Train vs Yhat_Train")
            selfPlotter.graph_plot (gd.yhat_test, gd.y_test, "Actual Values", "Estimated Values",
                                    "GD - Y_Test vs Yhat_Test ")
            selfPlotter.weight_graph_plot(gd.w_hat, "Features", "Weight of Features", "GD- Weights of Features", weight=1)

            """########################################################################################################################################"""
            """STEEPEST DESCENT"""


            """Find the local optimum weights using Steepest descent(Newton's method)- Faster than gradient descent"""
            """Uses Hessian matrix(double derivative over the error in weights of the previous estimate) to reach the minimum soon"""
            parkinson.logger.info ("Calculating using Steepest Descent(Newton's method)....")
            """Stop error value- value identified from plotting the cost function value against no of iterations"""
            sd.steepestDescent_Hessian (stopping_condn=10**-8, loop_limit=10**6)

            """Plot the progression of error values for the gradient descent algorithm"""
            parkinson.logger.info ("Plotting Steepest Descent....")
            selfPlotter.multiplot_sameX ('MSE_progression for Steepest descent', 'No.of Iterations', 'Change in E[w]'
            , ChangeInError_Training = sd.error_train, ChangeInError_Testing = sd.error_test)
            #selfPlotter.multiplot_sameX('Parameter change for Steepest descent','No.of Iterations','Change in W',ChangeInW=sd.w_vector)

            ##def multiplot_subplots_withXaxis(self,title, sp_rows, sp_cols, xlabel, ylabel, valuesPerSubplot, *xAxisValue,##**yaxisTitleAndValue)
            # selfPlotter.multiplot_subplots_withXaxis('Steepest descent - W vs d/dw J(w)', 1, 1, 'W', 'derivative of cost function', 1,sd.w_vector,
            #                                  steepest_descent = sd.cost_function_gradient_vector)
            """Steepest descent - the histogram of the error- comparing the error values for test and training data"""
            #multiplot_subHists(self, sp_rows, sp_cols, xlabel, ylabel, valuesPerSubplot, *bins, **yaxisTitleAndValue)
            selfPlotter.multiplot_subHists('Steepest Descent- Training error vs Testing error',1,2,'SD-error values','SD-error count',1, 50,50,
                                           training =sd.error_train_diff,testing=sd.error_test_diff)

            selfPlotter.graph_plot (sd.yhat_train, sd.y_train, "Actual Values", "Estimated Values",
                                    "SD - Y_Train vs Yhat_Train")
            selfPlotter.graph_plot (sd.yhat_test, sd.y_test, "Actual Values", "Estimated Values",
                                    "SD - Y_Test vs Yhat_Test ")

            selfPlotter.weight_graph_plot(sd.w_hat, "Features", "Weight of Features", "sd- Weights of Features", weight=1)


            # </editor-fold>

            # """########################################################################################################################################"""
            # """RIDGE REGRESSION"""
            # """Solves the problem of overfitting to training data by reducing the value of the W parameters through regularization"""
            # """Regularization - Penalising higher values of W by adding it to the cost function, with parameter lambda"""
            # """Lambda too low- normal Least square regression(may have overfitting), Lambda too low- Underfitting due too much bias"""
            # """Need to find the optimal lambda for the best model- Use K-fold validation for this"""
            # """In this program, I've used the variance between the MSE of the training and testing data as a measure to choose lambbda-
            # Lower this value, then there is less overfitting to the training data"""
            #
            # rr.ridgeRegression (init_lambda=0,max_lamda=200,lambda_step=0.5,kfold_splits=5)
            #
            # """Plot the progression of error values for the ridge regression algorithm"""
            # parkinson.logger.info ("Plotting RIDGE Regression....")
            # selfPlotter.multiplot_withXvalues ('Error_progression for RIDGE Regression', 'Lambda_Value', 'Change in E[w]',rr.lambda_values,rr.lambda_values,
            #                              ChangeInErrorTrain=rr.error_train,ChangeInErrorTest=rr.error_test)
            # selfPlotter.multiplot_withXvalues ('Error_Variance_progression for RIDGE Regression', 'Lambda_Value', 'Change in variance of E[w] - Test vs Train',rr.lambda_values,
            #                              trainVSTestError=rr.error_variance_testVStrain)
            #
            # ##def multiplot_subplots_withXaxis(self, title, sp_rows, sp_cols, xlabel, ylabel, valuesPerSubplot, *xAxisValue,##**yaxisTitleAndValue)
            # selfPlotter.multiplot_subplots_withXaxis('Ridge regression - Lambda vs W',1, 1, 'Lambda value', 'W value', 1, rr.lambda_values,
            #                                          w_values=rr.w_vector)
            #
            # ##def multiplot_subplots_withXaxis(self, title,sp_rows, sp_cols, xlabel, ylabel, valuesPerSubplot, *xAxisValue,##**yaxisTitleAndValue)
            # selfPlotter.multiplot_subplots_withXaxis ('Ridge regression - W vs error',1, 1, 'W value', 'E[W]', 1, rr.w_vector,
            #                                           ridge_regression_error=rr.error_test)
            #
            # """RIDGE regression - the histogram of the error- comparing ther error values for test and training data"""
            # #multiplot_subHists(self, sp_rows, sp_cols, xlabel, ylabel, valuesPerSubplot, *bins, **yaxisTitleAndValue)
            # selfPlotter.multiplot_subHists('Ridge regression- Training error vs Testing error',1,2,'RR-error values','RR-error count',1, 50,50,
            #                                training =rr.error_train,testing=rr.error_test)

            """########################################################################################################################################"""
            """COMPARING ALL FOUR ESTIMATION METHODS"""


            """Plot actual testing target vs estimated testing target for the four different methods"""
            """Comparing the estimates of the training data for each model"""
            # --yhat_train versus y_train
            # selfPlotter.multiplot_sameX('yhat_train vs y_train', 'features',target_field,
            #                             originalData=parkinson.y_train, MSE=mse.yhat_train, Gradient=gd.yhat_train, Steepest_descent=sd.yhat_train,
            #                             ridge_regression=rr.yhat_train)

            """Comparing the estimates of the testing data for each model"""
            # --yhat_test versus y_test
            selfPlotter.multiplot_sameX('yhat_test vs y_test', 'features',target_field,
                                        originalData=parkinson.y_test, MSE=mse.yhat_test, Gradient=gd.yhat_test, Steepest_descent=sd.yhat_test)

            """########################################################################################################################################"""


        except (ArithmeticError, OverflowError,FloatingPointError,ZeroDivisionError) as mathError:
            parkinson.logger.error('Mathematical Failure at main', exc_info=True)
            sys.exit()
        except (SystemExit, KeyboardInterrupt):
            sys.exit()
        except Exception as e:
            parkinson.logger.error('Common failure at main', exc_info=True)
            sys.exit()
        sys.exit (0)

