import sys
import pandas as pd
import logging
import logging.config
import os
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import subprocess as sp

class KidneyDisease(Exception):

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.path = os.getcwd ( )
        log_path = ('log_config.json')
        if os.path.exists(log_path):
            with open(log_path, 'rt') as f:
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



    def make_image(self,input_name, output_name):
        print("Input name:",input_name)
        print ("output_name name:", output_name)
        png = sp.run(['dot', '-Tpng', input_name, '-o', output_name], stdout=sp.PIPE,shell=True)
        print(png.stdout.decode('utf-8'))



    def readAndPrepCSVdata(self):
        try:
            """Read the data from the csv  file"""
            self.feature_Names = ['age', 'blood', 'specific gravity', 'albumin', 'sugar', 'red blood cells', 'pus cell', 'pus cell clumps',
                             'bacteria', 'blood glucose random', 'blood urea', 'serum creatinine', 'sodium', 'potassium',
                             'hemoglobin', 'packed cell volume', 'white blood cell count', 'red blood cell count',
                             'hypertension', 'diabetes mellitus', 'coronary artery disease', 'appetite', 'pedal edema',
                             'anemia', 'class']

            self.data = pd.read_csv (self.path+"\Data Source\chronic_kidney_disease.arff", header=None, skipinitialspace=True, skiprows=29,
                             na_values=['?', '\t?'],names=self.feature_Names, sep=',' )

            # Approach 2 - replace NaN Values

            self.data_replace_1 = self.data.fillna (-1)
            self.data_replace_2 = self.data.fillna (999999)

            #Approach 1 - drop NaN Values
            self.data_drop = self.data.dropna()

        except IOError:
            self.logger.error ('Failed to read file', exc_info=True)
            sys.exit ( )



        try:



            # Get only the necessary columns- Target only the result column and domain for all features
            self.data_drop_domain = self.data_drop.loc[:, :'anemia']
            self.data_drop_target = self.data_drop.loc[:, 'class']

            self.data_replace1_domain = self.data_replace_1.loc[:, :'anemia']
            self.data_replace1_target = self.data_replace_1.loc[:, 'class']

            self.data_replace2_domain = self.data_replace_2.loc[:, :'anemia']
            self.data_replace2_target = self.data_replace_2.loc[:, 'class']






            #We are going to use the decison tree regressor.
            #Decision treee regressor being used here(SciKit learn in python) supports only Numerical data
            #Replace categorical data with Numerical data
            # normal/notpresent/no/poor/notckd => 0
            # abnormal/present/yes/good/ckd => 1
            to_be_replaced = ['yes', 'no', '\tyes', '\tno', 'present', 'notpresent', 'abnormal', 'normal', 'good', 'poor','ckd','notckd','ckd\t','notckd\t','unknown']
            to_replace = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0,1,0,1,0,-1]




            self.data_drop_target = self.data_drop_target.replace (to_be_replaced, to_replace)
            self.data_replace1_target = self.data_replace1_target.replace (to_be_replaced, to_replace)
            self.data_replace2_target = self.data_replace2_target.replace (to_be_replaced, to_replace)
            self.data_drop_domain = self.data_drop_domain.replace (to_be_replaced, to_replace)
            self.data_replace1_domain = self.data_replace1_domain.replace (to_be_replaced, to_replace)
            self.data_replace2_domain = self.data_replace2_domain.replace (to_be_replaced, to_replace)


            clf = DecisionTreeClassifier ('entropy')




            clf_tree_drop = clf.fit (self.data_drop_domain, self.data_drop_target)



            print(self.path)

            self.class_names_drop = ['ckd', 'notckd']

            self.class_names_replace_1 = ['ckd', 'notckd', '1']
            self.class_names_replace_2 = ['ckd', 'notckd', '999999']
            self.class_names_replace = ['ckd', 'notckd', 'unknown']

            dot_data_drop = export_graphviz (clf_tree_drop, out_file=self.path+"\Tree_drop_final.dot",
                                             feature_names=self.feature_Names[0:24],
                                             class_names=self.class_names_drop, filled=True, rounded=True,
                                             special_characters=True)

            self.make_image (self.path+"\Tree_drop_final.dot", self.path+'\Tree_drop_final.png')

            feature_importance_drop = clf.feature_importances_

            plt.figure ( )
            plt.stem (feature_importance_drop)
            plt.xlabel ("Features")
            plt.ylabel ("Importance")
            plt.title ("Importance of Features - Scenario 1")
            plt.margins (0.01, 0.1)
            plt.grid ( )
            plt.show ( )

            clf_tree_replace1 = clf.fit (self.data_replace1_domain, self.data_replace1_target)


            dot_data_replace = export_graphviz (clf_tree_replace1, out_file=self.path+"\Tree_replace_final.dot",
                                                feature_names=self.feature_Names[0:24],
                                                class_names=self.class_names_replace_1, filled=True, rounded=True,
                                                special_characters=True)

            self.make_image (self.path+'\Tree_replace_final.dot', self.path+'\Tree_replace_final.png')

            feature_importance_replace = clf.feature_importances_


            plt.figure ( )
            plt.stem (feature_importance_replace)
            plt.xlabel ("Features")
            plt.ylabel ("Importance")
            plt.title ("Importance of Features - Scenario 2 - Negative Value")
            plt.margins (0.01, 0.1)
            plt.grid ( )
            plt.show()

            clf_tree_replace_2 = clf.fit (self.data_replace2_domain, self.data_replace2_target)

            dot_data_replace = export_graphviz (clf_tree_replace_2, out_file=self.path + "\Tree_replace_final.dot",
                                                feature_names=self.feature_Names[0:24],
                                                class_names=self.class_names_replace, filled=True, rounded=True,
                                                special_characters=True)

            self.make_image (self.path + '\Tree_replace_final.dot', self.path + '\Tree_replace_final.png')

            feature_importance_replace_2 = clf.feature_importances_

            plt.figure ( )
            plt.stem (feature_importance_replace_2)
            plt.xlabel ("Features")
            plt.ylabel ("Importance")
            plt.title ("Importance of Features - Scenario 2 - High Value")
            plt.margins (0.01, 0.1)
            plt.grid ( )
            plt.show ( )



        except (ArithmeticError, OverflowError, FloatingPointError, ZeroDivisionError) as mathError:
            self.logger.error ('Math error - Failed to prepare the csv data', exc_info=True)
            raise
        except Exception as e:
            self.logger.error ('Failed to prepare the csv data', exc_info=True)
            raise

if __name__ == "__main__":

        kidneyDisease = KidneyDisease()
        kidneyDisease.readAndPrepCSVdata()