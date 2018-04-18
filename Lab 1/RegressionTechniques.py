import random
import numpy as np
import logging
import os
import json

class RegressionTechniques(Exception):
    def __init__(self,x_train,x_test,y_train,y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
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
            self.error_train = np.square(self.yhat_train - self.y_train)
            self.mse_error_train = np.square(self.yhat_train-self.y_train).mean(axis=mean_axis)



            """Test data - apply the co-eff obtained from training to validate it against the test data"""
            self.logger.debug("Apply co-eff from traing to test data...")
            self.yhat_test = self.x_test.dot(self.w)
            self.error_test = np.square(self.yhat_test - self.y_test)
            self.mse_error_test= np.square(self.yhat_test-self.y_test).mean(axis=mean_axis)
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
        self.error = np.zeros(1)
        self.w_vector = np.zeros(1)


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

                """Calculating the error for the parameter values at iteration  """

                self.current_error = np.square (self.yhat_train - self.y_train).mean (axis=0)
                if (i == 0):
                    np.put(self.error, i, self.current_error)
                    np.put(self.w_vector,i,np.linalg.norm(self.w_hat))

                elif (i > 0):
                    self.error = np.append(self.error, self.current_error)
                    self.w_vector = np.append (self.w_vector, np.linalg.norm(self.w_hat))

                    if (abs(self.error[i-1] - self.current_error) <= stopping_condn or i >= loop_limit-1):
                        self.logger.info("GD- Final Change in Error:"+str((self.error[i-1] - self.current_error)))
                        self.logger.info ("GD- Final Change in W:" + str ((self.w_vector[i-1] - self.w_vector[i])))
                        self.logger.info("Gradient Descent - No.of iterations required to reach stopping condn:"+str(i))
                        self.logger.info("Gradient Descent - Value of error at stopping condn:" + str(self.current_error))
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

        return self.w_hat,self.gradient,self.error,self.current_error


    def steepestDescent_Hessian(self,stopping_condn,loop_limit):
        # %%SOLUTION 3 ___Steepest descnet algorithm a.k.a Newton's method using Hessian matrix
        # mu=0, sigma= 0.1, size= 21 - 21 FEATURES. Random gaussian values

        random.seed(40)
        init_w_hat = np.random.normal(0, 1, 21)

        self.w_hat = np.array(init_w_hat)

        self.hessian_matrix = self.calculate_Hessian()

        i = 0
        self.error = np.zeros(1)
        self.w_vector = np.zeros (1)

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

                """Calculating the error for the parameter values at iteration  """

                self.current_error = np.square (self.yhat_train - self.y_train).mean (axis=0)
                if (i == 0):
                    np.put(self.error, i, self.current_error)
                    np.put(self.w_vector,i,np.linalg.norm(self.w_hat))

                elif (i > 0):
                    self.error = np.append(self.error, self.current_error)
                    self.w_vector = np.append (self.w_vector, np.linalg.norm(self.w_hat))

                    if (abs(self.error[i-1] - self.current_error) <= stopping_condn or i >= loop_limit-1):
                        self.logger.info("Steepest Descent- Final Change in Error:"+str((self.error[i-1] - self.current_error)))
                        self.logger.info ("Steepest Descent- Final Change in W:" + str ((self.w_vector[i-1] - self.w_vector[i])))
                        self.logger.info("Steepest Descent - No.of iterations required to reach stopping condn:"+str(i))
                        self.logger.info("Steepest Descent - Value of error at stopping condn:" + str(self.current_error))
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

        return self.w_hat,self.gradient,self.error,self.current_error






