import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

# Read the cleaned and sorted Arrythmia file
data_csv = pd.read_csv ('DataSource/arrhythmia_cleaned_sorted.csv', sep=',')


data_df = pd.DataFrame (data_csv)

# Divide into target and features
class_id_vector = data_df.iloc[:, -1]
class_id_vector = class_id_vector.values
features = data_df.iloc[:, 0:-1]
features = features.values
features_copy = features.copy ( )
class_id_vector_copy = class_id_vector.copy ( )


features_submatrices = []
features_mean_for_each_class = []

for i in range (1, 17):
    if i == 11 or i == 12 or i == 13:
        features_submatrices.append (np.zeros ((1, 257)))
        features_mean_for_each_class.append(np.zeros ((1, 257)))

    else:
        separator_start = int (np.where (class_id_vector_copy == i)[0][0])

        separator_end = int (np.where (class_id_vector_copy == i)[0][-1])

        #print ("Seperator start:", separator_start)
        #print("Seperator end:",separator_end)
        features_submatrices.append (features_copy[separator_start:separator_end, :])
        #features_copy = features_copy[separator:, :]
        #class_id_vector_copy = class_id_vector_copy[separator:]


        features_mean_for_each_class.append((np.mean (features_submatrices[i - 1], axis=0)))

# %% min distance


diff_i = []
for i in range (0, len (features_mean_for_each_class)):
     diff_i.append (np.subtract (features, features_mean_for_each_class[i]))

dist_i = []
for i in range (0, len (features_mean_for_each_class)):
    dist_i.append (np.sum (np.square (diff_i[i]), axis=1))

est_class_id = []
for i in range (len (class_id_vector)):
    min_dist = 9999999
    class_est = -1
    #print("Actual class:",class_id_vector[i])
    for j in range (len (features_mean_for_each_class)):
        if dist_i[j][i] < min_dist and dist_i[j][i] != 0:
            min_dist = dist_i[j][i]
            class_est = j +1
    #print ("Estimated class:", class_est)
    est_class_id.append (class_est)

est_class_id = np.asarray (est_class_id)

# %% Confusion matrix
#    In case of multiclass classification, the confusion matrix
# is measured: the element in position i; j of the matrix is the probability
# that the estimated class is j given that the true class is i (the sum of the
# elements in a row must be 1).

correct_pred = 0

conf_matrix = np.zeros ((16, 16))
for k in range (class_id_vector.shape[0]):
    i = class_id_vector[k]
    j = est_class_id[k]
    conf_matrix[int (i) - 1][int (j) - 1] += 1
    if i == j:
        correct_pred += 1

for i in range (16):
    total = conf_matrix[i].sum ( )
    conf_matrix[i] = conf_matrix[i] / total

conf_matrix = np.nan_to_num (conf_matrix)

print("correct_pred:",correct_pred)
print("correct%:",(correct_pred/class_id_vector.shape[0])*100)

plt.figure (figsize=(12, 10))
sn.set (font_scale=1.4)  # for label size
sn.heatmap (conf_matrix, annot=True, annot_kws={"size": 12}, cmap="Blues", xticklabels=[i for i in range (1, 17)],
            yticklabels=[i for i in range (1, 17)])  # font size
plt.title ("Confusion matrix of Minimum distance criteria with 16 classes")
plt.xlabel ("Estimated Class")
plt.ylabel ("Actual Class")
plt.show ( )
plt.savefig ('Min distance 16')


