# -*- coding: utf-8 -*-
"""
Created on  Jul 29 14:24:12 2018

@author: Deepan
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

seed = 42

file_name_list_lr = ["low_risk_"+str(x) for x in range (1, 12)]
file_name_list_mr = ["medium_risk_"+str(x) for x in range (1, 17)]
file_name_list_melanoma = ["melanoma_"+str(x) for x in range (1, 28)]

file_name_list = file_name_list_lr.append(file_name_list_mr,file_name_list_melanoma)

ratio_lr = []
ratio_mr = []
ratio_hr = []

t_ratio = []
t_perimeter_m = []
t_area_m = []
t_perimeter_c = []


for file_name in file_name_list:
    print ("File being processed:",file_name)

    file_path = 'Datasource/' + file_name + '.jpg'

    im = mpimg.imread (file_path)

    [N1, N2, N3] = im.shape
    im_2D = im.reshape ((N1 * N2, N3))

    kmeans = KMeans (n_clusters=3, random_state=seed)
    kmeans.fit (im_2D)

    centroids = kmeans.cluster_centers_.astype ('uint8')
    labels = kmeans.labels_

    im_3D = kmeans.predict (im_2D).reshape ((N1, N2))

    colors = []
    for i in centroids:
        colors.append (i.sum ( ))
    darkest_color = colors.index (min (colors))
    darkest_color = np.unique (centroids.argmin (axis=0))[0]

    plt.figure ( )
    plt.imshow (im)

    plt.figure ( )
    plt.imshow (im_3D)
    plt.savefig ("Output/" + file_name + "_kmeans.png")

    # %% zoom
    if file_name == "melanoma_23":
        drop = 50
        row, column = im_3D.shape
        im_3D = im_3D[0:row - drop, drop:column]
        plt.figure ( )
        plt.imshow (im_3D)

    # %% find center
    middle_N1 = int (N1 / 2)
    middle_N2 = int (N2 / 2)

    count_h = np.count_nonzero (im_3D[middle_N1, :] == darkest_color)

    temp = 0
    for i in range (N2):
        if im_3D[middle_N1, i] == darkest_color:
            temp += 1
        if temp == int (count_h / 2):
            center_h = i
            break

    count_v = np.count_nonzero (im_3D[:, middle_N2] == darkest_color)

    temp = 0
    for i in range (N1):
        if im_3D[i, middle_N2] == darkest_color:
            temp += 1
        if temp == int (count_v / 2):
            center_v = i
            break

    # %% crop
    new_im_3D = im_3D
    i = center_v
    while i >= 0:
        pixel_y = np.count_nonzero (new_im_3D[i, :] == darkest_color)
        if pixel_y > 0:
            i -= 1
        elif pixel_y == 0:
            crop_sup = i + 1
            break

    i = center_v
    while i < N1:
        pixel_y = np.count_nonzero (new_im_3D[i, :] == darkest_color)
        if pixel_y > 0:
            i += 1
        elif pixel_y == 0:
            crop_inf = i - 1
            break

    i = center_h
    while i >= 0:
        pixel_y = np.count_nonzero (new_im_3D[:, i] == darkest_color)
        if pixel_y > 0:
            i -= 1
        elif pixel_y == 0:
            crop_left = i + 1
            break

    i = center_h
    while i < N2:
        pixel_y = np.count_nonzero (new_im_3D[:, i] == darkest_color)
        if pixel_y > 0:
            i += 1
        elif pixel_y == 0:
            crop_right = i - 1
            break

    new_im_3D = new_im_3D[crop_sup:crop_inf, crop_left:crop_right]
    plt.figure ( )
    plt.imshow (new_im_3D)

    # %% filter
    new_im_3D[new_im_3D != darkest_color] = 0
    [M1, M2] = new_im_3D.shape
    for k in range (2):
        for i in range (M1):
            for j in range (M2):
                if i >= 1 and i < M1 - 1 and j >= 1 and j < M2 - 1:
                    neighbors = new_im_3D[i - 1, j] + new_im_3D[i + 1, j] + new_im_3D[i, j - 1] + new_im_3D[i, j + 1]
                    if neighbors < 2 * darkest_color:
                        new_im_3D[i, j] = 0
                    elif neighbors > 2 * darkest_color and new_im_3D[i, j] != darkest_color:
                        new_im_3D[i, j] = darkest_color

    plt.figure ( )
    plt.imshow (new_im_3D)
    plt.savefig ("Output/" + file_name + "_crop.png")

    # %% contour columns
    min_col = []
    max_col = []
    [M1, M2] = new_im_3D.shape
    for i in range (M2):
        for j in range (M1):
            if i < M2 - 1:
                if new_im_3D[j, i] == darkest_color and new_im_3D[j, i + 1] == darkest_color:
                    min_col.append ((j, i))
                    break
            else:
                if new_im_3D[j, i] == darkest_color:
                    min_col.append ((j, i))
                    break

    for i in range (M2):
        for j in reversed (range (M1)):
            if i >= 1:
                if new_im_3D[j, i] == darkest_color and new_im_3D[j, i - 1] == darkest_color:
                    max_col.append ((j, i))
                    break
            else:
                if new_im_3D[j, i] == darkest_color:
                    max_col.append ((j, i))
                    break

    # %% contour rows
    min_row = []
    max_row = []
    for i in range (M1):
        for j in range (M2):
            if i < M1 - 1:
                if new_im_3D[i, j] == darkest_color and new_im_3D[i + 1, j] == darkest_color:
                    min_row.append ((i, j))
                    break
            else:
                if new_im_3D[i, j] == darkest_color:
                    min_row.append ((i, j))
                    break

    for i in range (M1):
        for j in reversed (range (M2)):
            if i >= 1:
                if new_im_3D[i, j] == darkest_color and new_im_3D[i - 1, j] == darkest_color:
                    max_row.append ((i, j))
                    break
            else:
                if new_im_3D[i, j] == darkest_color:
                    max_row.append ((i, j))
                    break

    # %% plot contour
    contour = np.zeros ((M1, M2))
    list_contour = min_col + max_col + min_row + max_row
    for i in list_contour:
        contour[i] = 1

    plt.matshow (contour, cmap="Blues")
    plt.savefig ("Output/" + file_name + "_contour.png")

    # %% area and perimeter
    area_m = np.count_nonzero (new_im_3D == darkest_color)
    perimeter_m = np.count_nonzero (contour == 1)

    radio_circ = np.sqrt (area_m / np.pi)
    perimeter_circ = 2 * np.pi * radio_circ

    ratio = perimeter_m / perimeter_circ

    if "low_risk" in file_name:
        ratio_lr.append (ratio)

    elif "medium_risk" in file_name:
        ratio_mr.append (ratio)

    else:
        ratio_hr.append (ratio)

    plt.close ("all")

    # %% table
    t_ratio.append (ratio)
    t_perimeter_m.append (perimeter_m)
    t_area_m.append (area_m)
    t_perimeter_c.append (perimeter_circ)

# %% plot
plt.figure ( )
lr = plt.plot (ratio_lr, color='red', label='Low Risk')
mr = plt.plot (ratio_mr, color='blue', label='Medium Risk')
hr = plt.plot (ratio_hr, color='green', label='Melanoma')
plt.xlabel ("Picture")
plt.ylabel ("Ratio")
plt.title ("Ratio between perimeter of the mole and perimeter of the circle with the same area")
plt.grid ( )
plt.legend ( )
plt.show ( )

print ("Low risk:", np.mean (ratio_lr))
print ("Medium risk:", np.mean (ratio_mr))
print ("Melanoma:", np.mean (ratio_hr))
