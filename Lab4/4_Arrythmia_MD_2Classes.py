#Lab6: Classification part 1: Minimum distance, 2 classes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the file and put NaN instead of ?
ar_data_original = pd.read_csv ('DataSource/arrhythmia.data.csv', sep=',', header=None, na_values=['? ','?\t','?', '\t?'])

ar_data_copy = ar_data_original.copy ( )
# Drop the rows with Nan values (axis=1 is columns):
ar_data_copy = ar_data_copy.dropna (axis=1)

# Remove the Columns that are all 0 - zero_columns is a column vector that has the summation along each column
ar_data_copy_values = ar_data_copy.values
zero_columns = np.sum (np.abs (ar_data_copy), axis=0)
zero_column_indices = np.where (zero_columns == 0)
data = np.delete (ar_data_copy_values, zero_column_indices, axis=1)

# Sort data- data_df.shape give no.of rows, no.of columns - we sort data on the last column - CLASSES
data_df = pd.DataFrame (data)
data_df = data_df.sort_values (by=[(data_df.shape[1]-1)])
data_sorted = data_df.values

#Saving the cleaned and sorted data as csv

data_df.to_csv('DataSource/arrhythmia_cleaned_sorted.csv', sep=',', encoding='utf-8' )

# target
target = data_sorted[:, -1]

# go from 16 classes to 2 classes. Previous class1 is class 1. Previous classes
# 2 to 16 is now Class 2
new_targets = 1 * (target == 1) + 2 * (target > 1)
Y = data_sorted[:, 0:-1]

plt.hist (new_targets)

class_id = new_targets
# Define the two submatrices: y1, with the rows/patients corresponding to class id=1,
# #and y2, with the rows/patients corresponding to class id=2
last_index_Y1 = int (np.where (class_id == 1)[0][-1]) + 1
y1 = Y[0:last_index_Y1, :]
y2 = Y[last_index_Y1:, :]

x1 = np.mean (y1, axis=0)
x2 = np.mean (y2, axis=0)

# Minimum distance criterion
diff_class1 = np.subtract (Y, x1)
diff_class2 = np.subtract (Y, x2)
dist_class1 = np.sum (np.square (diff_class1), axis=1)
dist_class2 = np.sum (np.square (diff_class2), axis=1)
dist = np.subtract (dist_class1, dist_class2)

est_class_id = np.empty (dist.shape[0])
est_class_id[np.where (dist > 0)] = 2
est_class_id[np.where (dist < 0)] = 1

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

print ("True positive:", true_positive)
print ("True negative:", true_negative)
print ("False positive:", false_positive)
print ("False negative:", false_negative)
print ("Sensitivity:", sensitivity)
print ("Specificity:", specificity)



