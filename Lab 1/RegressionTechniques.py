import random
import numpy as np
import logging
import os
from sklearn.model_selection import KFold
import json

class RegressionTechniques(Exception):
    def __init__(self,x_train,x_test,y_train,y_test,data):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.data = data
        self.x_train_transpose = x_train.transpose()

        self.logger = logging.getLogger(__name__)
        path = ('log_config.json')
        if os.path.exists(path):
            with open(path, 'rt') as f:
                config = json.load(f)
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


    def calculate_Hessian(self):
        return (np.multiply (self.x_train_transpose.dot (self.x_train), 4))

    def calcaulate_gradient(self,init_w_hat):

        return (np.multiply ((self.x_train_transpose.dot (self.y_train)), -2) + np.multiply (
            ((self.x_train_transpose.dot (self.x_train)).dot (init_w_hat)), 2))

    def calculate_ridge_estimate(self,x_train_data,y_train_data,lambda_penalty):

        size = np.shape (x_train_data)
        identity = np.identity(size[1])

        temp_xTXinv_ridge = (np.linalg.inv(x_train_data.T.dot(x_train_data) + lambda_penalty * identity))
        self.w_hat = (temp_xTXinv_ridge.dot(x_train_data.T)).dot(y_train_data)
        return self.w_hat





    def calculate_MSE(self,mean_axis=0):

        """#%%------MSE
        #w -weight of parameter to be estimated
        #X -input features for the different parameters that affect the Target
        #Y - Actual measured output
        #Yhat - Estimated value of output/target"""
        #Mean axis value decides how we calculate the error for the estimate
        #If means_axis = 0, mean is taken by adding the values along the rows and then dividing-result matches the no.of.cols of data
        #If means_axis = 1, mean is taken by adding the values along the columns and then dividing -result matched no.of rows of data
        #If means_axis = None, mean is taken elementwise - returns a single value

        """"# w= [X^t*X]^-1*X^t*y but [X^t*X]^-1*X^t -> pseudo-inverse of the rectangular matrix X
        Therefore w=(pseudo-inverse of the rectangular matrix X)y"""

        try:
            #----NORMALIZE DATA
            #-Train data
            """Calculate coeficients for different features"""
            self.logger.debug("Calculate coeficients for different features...")
            self.w=np.dot((np.linalg.pinv(self.x_train)),self.y_train)
            """ Get the estimated value based on the co-efficients and features"""
            self.logger.debug("Getting MSE Estimate...")
            self.yhat_train = self.x_train.dot(self.w)
            """Calculate the mean squared error"""
            self.logger.debug("Getting MSE Error...")
            self.error_train_MSE = np.linalg.norm(self.yhat_train-self.y_train)**2


            """Test data - apply the co-eff obtained from training to validate it against the test data"""
            self.logger.debug("Apply co-eff from traing to test data...")
            self.yhat_test = self.x_test.dot(self.w)
            self.error_test_mse= np.linalg.norm(self.yhat_test-self.y_test)**2
            self.error_train = self.y_train - self.yhat_train
            self.error_test = self.y_test - self.yhat_test
            self.logger.info("MSE-Final error value:"+str(self.error_test))
        except (ArithmeticError, OverflowError, FloatingPointError, ZeroDivisionError) as mathError:
            self.logger.error('Math error - Failed to prepare the csv data', exc_info=True)
            raise
        except Exception as e:
            self.logger.error('Failed to calculate MSE', exc_info=True)
            raise

        """Comments---> We found the coefficients (w) once with the train data, and then find y_hat for both train and
                    test data with those coefficients. The estimated values doesn't follow the same behavior as the
                    original data because the choose of the UPDR values is not the same per each doctor, this values
                    are implicit so it's easy to have high variance and this size in the error btw the data."""

        return self.yhat_train,self.error_train,self.yhat_test,self.error_test

    def gradientDescent(self,stopping_condn,loop_limit,gamma):
        # %%------SOLUTION 2 ___Gradient algorithm
        # mu=0, sigma= 0.1, size= 21 - 21 FEATURES. Random gaussian values
        random.seed(40)
        init_w_hat = np.random.normal(0, 1, 21)

        self.w_hat = np.array(init_w_hat)


        i = 0
        self.error_train_GD = np.zeros(1)
        self.error_test_GD = np.zeros (1)
        self.error_train = np.zeros(1)
        self.error_test = np.zeros (1)
        self.w_vector = np.zeros(1)
        self.cost_function_gradient_vector = np.zeros(1)


        # re( ^w(i)) = -2XTy + 2XTX^w(i)

        # I have a stopping condition for the error value diff,
        # but to avoid infinite looping, i put an upper limit
        self.logger.debug("Start loop for gradient descent with stopping condn...")
        try:
            for i in range(loop_limit):
                # Evaluate the gradient

                prev_w_hat = self.w_hat
                """Gradient is the partial differentiation over w of th error term E[W]"""
                self.gradient = self.calcaulate_gradient(prev_w_hat)

                self.w_hat = (prev_w_hat - np.multiply(gamma, self.gradient))

                self.yhat_train = self.x_train.dot (self.w_hat)
                self.yhat_test = self.x_test.dot (self.w_hat)

                """Calculating the error for the parameter values at iteration  """

                self.current_error_test_GD = np.linalg.norm(self.yhat_test - self.y_test)**2
                self.current_error_train_GD = np.linalg.norm(self.yhat_train - self.y_train)**2

                self.current_error_test = (self.y_test - self.yhat_test)
                self.current_error_train = (self.y_train - self.yhat_train)
                if (i == 0):
                    np.put(self.error_train_GD, i, self.current_error_train_GD)
                    np.put (self.error_test_GD, i, self.current_error_test_GD)
                    np.put(self.error_train, i, self.current_error_train)
                    np.put (self.error_test, i, self.current_error_test)
                    np.put(self.w_vector,i,np.linalg.norm(self.w_hat))
                    np.put(self.cost_function_gradient_vector,i,self.gradient[0])


                elif (i > 0):
                    self.error_train_GD = np.append(self.error_train, self.current_error_train_GD)
                    self.error_test_GD = np.append (self.error_test, self.current_error_test_GD)
                    self.error_train = np.append(self.error_train, self.current_error_train)
                    self.error_test = np.append (self.error_test, self.current_error_test)
                    self.w_vector = np.append (self.w_vector, np.linalg.norm(self.w_hat))
                    self.cost_function_gradient_vector = np.append (self.cost_function_gradient_vector, self.gradient[0])


                    if (abs(self.w_vector[i-1] - self.w_vector[i]) <= stopping_condn or i >= loop_limit-1):
                        self.logger.info("GD- Final Change in Error:"+str((self.error_test[i-1] - self.current_error_test)))
                        self.logger.info ("GD- Final Change in W:" + str ((self.w_vector[i-1] - self.w_vector[i])))
                        self.logger.info("Gradient Descent - No.of iterations required to reach stopping condn:"+str(i))
                        self.logger.info("Gradient Descent - Value of error at stopping condn:" + str(self.current_error_test))
                        self.yhat_test = self.x_test.dot (self.w_hat)
                        self.yhat_train = self.x_train.dot (self.w_hat)

                        break
        except (SystemExit, KeyboardInterrupt):
            raise
        except (ArithmeticError, OverflowError,FloatingPointError,ZeroDivisionError) as mathError:
            self.logger.error('Math error - Failed to prepare the csv data', exc_info=True)
            raise

        except Exception as e:
            self.logger.error('Failed to calculate gradient descent', exc_info=True)
            raise

        return self.w_hat,self.gradient


    def steepestDescent_Hessian(self,stopping_condn,loop_limit):
        # %%SOLUTION 3 ___Steepest descnet algorithm a.k.a Newton's method using Hessian matrix
        # mu=0, sigma= 0.1, size= 21 - 21 FEATURES. Random gaussian values

        random.seed(40)
        init_w_hat = np.random.normal(0, 1, 21)

        self.w_hat = np.array(init_w_hat)

        self.hessian_matrix = self.calculate_Hessian()

        i = 0
        self.error_train = np.zeros(1)
        self.error_test = np.zeros (1)
        self.error_train_sd = np.zeros(1)
        self.error_test_sd = np.zeros (1)
        self.w_vector = np.zeros (1)
        self.cost_function_gradient_vector = np.zeros(1)

        # re( ^w(i)) = -2XTy + 2XTX^w(i)

        # I have a stopping condition for the error value diff,
        # but to avoid infinite looping, i put an upper limit
        self.logger.debug("Start loop for Steepest descent/Newton method with stopping condn...")
        try:
            for i in range(loop_limit):
                # Evaluate the gradient

                prev_w_hat = self.w_hat

                """Gradient is the partial differentiation over w of th error term E[W]"""
                self.gradient = self.calcaulate_gradient(prev_w_hat)

                self.gradient_transpose = self.gradient.transpose()

                self.gradient_norm_squared = np.square(np.linalg.norm(self.gradient))

                self.numer_learning_rate = np.multiply(self.gradient ,self.gradient_norm_squared)

                self.denom_learning_rate = (self.gradient_transpose.dot(self.hessian_matrix)).dot(self.gradient)

                self.w_hat = (prev_w_hat - np.divide(self.numer_learning_rate, self.denom_learning_rate))

                self.yhat_train = self.x_train.dot (self.w_hat)
                self.yhat_test = self.x_test.dot (self.w_hat)

                """Calculating the error for the parameter values at iteration  """

                self.current_error_test_sd = np.linalg.norm(self.yhat_test - self.y_test)**2
                self.current_error_train_sd = np.linalg.norm(self.yhat_train - self.y_train)**2

                self.current_error_test = (self.y_test - self.yhat_test)
                self.current_error_train = (self.y_train - self.yhat_train)
                if (i == 0):
                    np.put(self.error_train_sd, i, self.current_error_train_sd)
                    np.put (self.error_test_sd, i, self.current_error_test_sd)
                    np.put(self.error_train, i, self.current_error_train)
                    np.put (self.error_test, i, self.current_error_test)
                    np.put(self.w_vector,i,np.linalg.norm(self.w_hat))
                    np.put(self.cost_function_gradient_vector,i,self.gradient[0])


                elif (i > 0):
                    self.error_train_sd = np.append(self.error_train, self.current_error_train_sd)
                    self.error_test_sd = np.append (self.error_test, self.current_error_test_sd)
                    self.error_train = np.append(self.error_train, self.current_error_train)
                    self.error_test = np.append (self.error_test, self.current_error_test)
                    self.w_vector = np.append (self.w_vector, np.linalg.norm(self.w_hat))
                    self.cost_function_gradient_vector = np.append (self.cost_function_gradient_vector, np.linalg.norm(self.gradient))

                    if (abs(self.w_vector[i-1] - self.w_vector[i]) <= stopping_condn or i >= loop_limit-1):
                        self.logger.info("Steepest Descent- Final Change in Error:"+str((self.error_test[i-1] - self.current_error_test)))
                        self.logger.info ("Steepest Descent- Final Change in W:" + str ((self.w_vector[i-1] - self.w_vector[i])))
                        self.logger.info("Steepest Descent - No.of iterations required to reach stopping condn:"+str(i))
                        self.logger.info("Steepest Descent - Value of error at stopping condn:" + str(self.current_error_test))
                        self.yhat_test = self.x_test.dot (self.w_hat)
                        self.yhat_train = self.x_train.dot (self.w_hat)

                        break
        except (SystemExit, KeyboardInterrupt):
            raise
        except (ArithmeticError, OverflowError,FloatingPointError,ZeroDivisionError) as mathError:
            self.logger.error('Math error - Failed to prepare the csv data', exc_info=True)
            raise

        except Exception as e:
            self.logger.error('Failed to calculate gradient descent', exc_info=True)
            raise

        return self.w_hat,self.gradient

    def ridgeRegression(self,init_lambda,max_lamda,lambda_step,kfold_splits):
        # %%------SOLUTION 4 ___Gradient algorithm with regularisation - RIDGE regression
        # mu=0, sigma= 0.1, size= 21 - 21 FEATURES. Random gaussian values
        random.seed(400)
        init_w_hat = np.random.normal(0, 1, 21)

        self.w_hat = np.array(init_w_hat)


        i = 0
        self.error_train = np.zeros(1)
        self.error_test = np.zeros (1)
        self.error_train_RR = np.zeros(1)
        self.error_test_RR = np.zeros (1)
        self.w_vector = np.zeros(1)
        self.error_variance_testVStrain = np.zeros (1)
        self.lambda_values = np.zeros(1)
        #Set up the lambda values for regularisation
        #Loop until you find best lambda or stopping condn
        lambda_current = init_lambda
        best_lambda = lambda_current

        loop_limit = int((max_lamda-init_lambda)/lambda_step)
        prev_error_variance_testVStrain = 999999999999

        # re(w(i)) = inv (xTx +lamb(I)). xTy

        # I have a stopping condition for the number of diff lambda checks,

        """Combine the test and train data to enable K-fold splitting"""

        self.logger.debug ("Start loop for RIDGE regression")

        try:
            for i in range(loop_limit):


                """Calculate the w_estimate based on RIDGE regression equation"""

                kf = KFold (n_splits=kfold_splits)
                k = 0
                self.fold_error_train = np.zeros (1)
                self.fold_error_test = np.zeros (1)

                for train, test in kf.split (self.data):
                    train_data = np.array (self.data)[train]
                    test_data = np.array (self.data)[test]
                    """After combining test and train data-split target and domain based on k-fold split"""
                    train_data_split = np.hsplit (train_data, np.array ([-1]))
                    test_data_split = np.hsplit (test_data, np.array ([-1]))
                    #FO = "total_UPDRS"
                    #x_train_data = train_data[FO].values
                    #y_train_data = train_data.drop ([FO], axis=1).values
                    #x_test_data = test_data[FO].values
                    #y_test_data = test_data.drop ([FO], axis=1).values

                    x_train_data = train_data_split[0]
                    x_test_data = test_data_split[0]
                    y_train_data = train_data_split[1]
                    y_test_data = test_data_split[1]


                    self.w_hat = self.calculate_ridge_estimate(x_train_data,y_train_data,lambda_current)
                    yhat_test_data = x_test_data.dot (self.w_hat)
                    yhat_train_data = x_train_data.dot (self.w_hat)


                    """Calculating the error for the parameter values at iteration using test values """

                    current_fold_error_test = np.linalg.norm(yhat_test_data - y_test_data)**2
                    current_fold_error_train = np.linalg.norm(yhat_train_data - y_train_data)**2

                    if (k == 0):
                        np.put (self.fold_error_train, k, current_fold_error_train)
                        np.put (self.fold_error_test, k, current_fold_error_test)


                    elif (k > 0):
                        self.fold_error_train = np.append (self.fold_error_train, current_fold_error_train)
                        self.fold_error_test = np.append (self.fold_error_test, current_fold_error_test)
                    k = k+1

                """Check for the best lambda value by calculating the mean error of the k-folds for current lambda"""
                error_variance_current_lambda_testVStrain = (np.linalg.norm(self.fold_error_test -self.fold_error_train))**2

                self.mean_error_test_for_Current_lambda = self.fold_error_test.mean(axis=0)
                self.mean_error_train_for_Current_lambda = self.fold_error_train.mean (axis=0)

                error_variance_present_check = (self.mean_error_test_for_Current_lambda  - self.mean_error_train_for_Current_lambda)**2



                print ("lambda_current:", lambda_current)
                print ("error_variance_present_check:", error_variance_present_check)
                print("error_variance_current_lambda_testVStrain:",error_variance_current_lambda_testVStrain)



                if error_variance_current_lambda_testVStrain < prev_error_variance_testVStrain:
                    best_lambda = lambda_current
                    prev_error_variance_testVStrain = error_variance_current_lambda_testVStrain


                lambda_current = lambda_current + lambda_step

                if (i == 0):
                    np.put(self.error_test, i, self.mean_error_test_for_Current_lambda)
                    np.put (self.error_train, i, self.mean_error_train_for_Current_lambda)
                    np.put (self.error_variance_testVStrain, i, error_variance_current_lambda_testVStrain)
                    np.put(self.w_vector,i,np.linalg.norm(self.w_hat))
                    np.put (self.lambda_values, i, lambda_current)


                elif (i > 0):
                    self.error_test = np.append(self.error_test, self.mean_error_test_for_Current_lambda)
                    self.error_train = np.append (self.error_train, self.mean_error_train_for_Current_lambda)
                    self.error_variance_testVStrain = np.append (self.error_variance_testVStrain, error_variance_current_lambda_testVStrain)
                    self.w_vector = np.append (self.w_vector, np.linalg.norm(self.w_hat))
                    self.lambda_values = np.append (self.lambda_values, lambda_current)

            """Best lambda was found using K-fold, now use it calculate W using the generic x_train and y_train"""
            self.w_hat = self.calculate_ridge_estimate(self.x_train,self.y_train,best_lambda)

            self.yhat_train = self.x_train.dot (self.w_hat)
            self.yhat_test = self.x_test.dot (self.w_hat)
            final_error = np.linalg.norm(self.yhat_test - self.y_test)**2

            self.logger.info("Ridge Regression-best lambda value:"+str(best_lambda))
            self.logger.info ("Ridge Regression- Final error:" + str (final_error))





        except (SystemExit, KeyboardInterrupt):
            raise
        except (ArithmeticError, OverflowError,FloatingPointError,ZeroDivisionError) as mathError:
            self.logger.error('Math error - Failed to prepare the csv data', exc_info=True)
            raise

        except Exception as e:
            self.logger.error('Failed to calculate ridge regression', exc_info=True)
            raise

        return self.w_hat,best_lambda







