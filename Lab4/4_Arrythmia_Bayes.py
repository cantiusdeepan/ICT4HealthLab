
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from CustomPlotter import CustomPlotter1

def pcr(x_train, percent=0.99):
    X_train_rows, X_train_columns = x_train.shape
    R_Cov_X_train = np.cov(x_train.T)
    eig_values, eig_vectors = np.linalg.eig(R_Cov_X_train)
    eig_values = np.real(eig_values)
    eig_vectors = np.real (eig_vectors)

    P_param_eig_va_sum = np.absolute(eig_values).sum ( )
    ordered_eigen_values = np.flip (np.argsort (eig_values), 0)
    sum_eig_values = 0
    L_Important_features_count = 0
    keep_feature = []
    while sum_eig_values < percent * P_param_eig_va_sum and L_Important_features_count < len(ordered_eigen_values):
        sum_eig_values += abs(eig_values[ordered_eigen_values[L_Important_features_count]])
        keep_feature.append (ordered_eigen_values[L_Important_features_count])
        L_Important_features_count += 1
    print ("L_Important_features_count:",L_Important_features_count)

    features_to_delete = np.setdiff1d (range (len (eig_values)), keep_feature)
    eig_values_L = np.delete (eig_values, features_to_delete)

    #lambda_L = np.eye (L_Important_features_count) * eig_values_L
    eig_vectors_L = np.delete (eig_vectors, features_to_delete, 1)
    #temp = eig_vectors_L.dot (np.linalg.inv (lambda_L)) / X_train_rows

    #w = temp.dot (eig_vectors_L.T).dot (x_train.T).dot (y_train)

    return eig_values_L,eig_vectors_L,L_Important_features_count

# Read the cleaned and sorted Arrythmia file
data_csv = pd.read_csv ('DataSource/arrhythmia_cleaned_sorted.csv', sep=',')


data_df = pd.DataFrame (data_csv)

data2 = data_df.values


target = data2[:,-1]

#go from 16 classes to 2 classes. Previous class1 is class 1. Previous classes
#2 to 16 is now Class 2
t = 1*(target==1)+2*(target>1)
Y = data2[:, 0:-1]


class_id = t
#Define the two submatrices: y1, with the rows/patients corresponding to
#class id=1, and y2, with the rows/patients corresponding to
#class id=2
separator = int(np.where(t==1)[0][-1])+1
y1 = Y[0:separator,:]
y2 = Y[separator:,:]

x1=np.mean(y1,axis=0)
x2=np.mean(y2,axis=0)

#%% Bayes criterion
num_patients = len(class_id)
pi1 = np.count_nonzero(class_id == 1)/num_patients #without arrythmia/total no. of patients
pi2 = np.count_nonzero(class_id == 2)/num_patients #with arrythmia/total no. of patients


percents = [0.7,0.8,0.9,0.93,0.95,0.97,0.99,0.99,0.9999,0.999999,0.99999999]
specificity_list = []
sensitivity_list = []
selected_features_list_1 = []
selected_features_list_2 = []
percentages = []
R_Cov_X = np.cov (Y.T)
eig_values_orig, eig_vectors_orig = np.linalg.eig(R_Cov_X)
for percent in percents:
    eig_values_red1,eig_vector_red1,selected_features_1 = pcr(y1,percent)
    eig_values_red2,eig_vector_red2,selected_features_2 = pcr(y2,percent)

    # projection
    z1 = y1.dot(eig_vector_red1)
    z2 = y2.dot(eig_vector_red2)

    w1 = z1.mean(axis=0)
    w2 = z2.mean(axis=0)

    s1 = Y.dot(eig_vector_red1)
    s2 = Y.dot(eig_vector_red2)

    # pdf
    R1_red = np.cov(z1.T)
    R2_red =np.cov(z2.T)

    pdf1 = []  # healthy
    M, F = s1.shape
    for i in range (M):
        temp1 = 1 / np.sqrt (((2 * np.pi) ** F) * np.linalg.det (R1_red))
        temp2 = (-1 / 2) * ((s1[i, :] - w1).T).dot (np.linalg.inv (R1_red)).dot (s1[i, :] - w1)
        pdf1.append (temp1 * np.exp (temp2))

    pdf2 = []  # sick
    M, F = s2.shape
    for i in range (M):
        temp1 = 1 / np.sqrt (((2 * np.pi) ** F) * np.linalg.det (R2_red))
        temp2 = (-1 / 2) * ((s2[i, :] - w2).T).dot (np.linalg.inv (R2_red)).dot (s2[i, :] - w2)
        pdf2.append (temp1 * np.exp (temp2))

    est_class_id = []
    for i in range (num_patients):
        if pi1 * pdf1[i] > pi2 * pdf2[i]:
            est_class_id.append (1)
        else:
            est_class_id.append (2)

    # %% Find probabilities of true/false positives and true/false negatives
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for i in range (len (est_class_id)):
        if est_class_id[i] == 1:  # Tn
            if class_id[i] == 1:  # Dn
                true_negative += 1
            elif class_id[i] == 2:  # Dy
                false_negative += 1
        elif est_class_id[i] == 2:  # Tp
            if class_id[i] == 1:  # Dn
                false_positive += 1
            elif class_id[i] == 2:  # Dy
                true_positive += 1

    sensitivity = float (true_positive) / (true_positive + false_negative)
    specificity = float (true_negative) / (true_negative + false_positive)
    print("percent chosen:", percent)
    print ("True positive:", true_positive)
    print ("True negative:", true_negative)
    print ("False positive:", false_positive)
    print ("False negative:", false_negative)
    specificity_list.append(specificity)
    sensitivity_list.append(sensitivity)
    selected_features_list_1.append(selected_features_1)
    selected_features_list_2.append (selected_features_2)
    percentages.append(percent)



    print ("Sensitivity:", sensitivity)
    print ("Specificity:", specificity)

selfPlotter = CustomPlotter1()
selfPlotter1 = CustomPlotter1()

print("PCR - percentages:",percentages)
print("specificity_list:",specificity_list)
print("sensitivity_list:",sensitivity_list)
print("selected_features_list_1-Healthy:",selected_features_list_1)
print("selected_features_list_2-Sick:",selected_features_list_2)


selfPlotter.multiplot_subHists('EigenValues- Histogram',1,1,'Eigen Values','Count',1, 50,
                                           eigen_values =eig_values_orig)
#selfPlotter1.multiplot_withXvalues('Sensitivity and Specificity progression with PCR percentage','Percentage','Change in specificity & sensitivity',percentages,percentages,sensitivity_=sensitivity_list,specificity_=specificity_list)

#selfPlotter1.multiplot_withXvalues('Sensitivity and Specificity progression with PCR percentage','Percentage','Change in specificity & sensitivity',percents,percents,sensitivity_1=sensitivity_list,specificity_1=specificity_list)


