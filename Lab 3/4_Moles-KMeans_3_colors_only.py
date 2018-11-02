# -*- coding: utf-8 -*-
"""
Created on  Jul 29 14:24:12 2018

@author: Deepan
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


ratio_lr_first = []
ratio_mr_first = []
ratio_hr_first = []

ratio_lr_improved = []
ratio_mr_improved = []
ratio_hr_improved = []

ratio_first_all = []
ratio_improved_all = []
perimeter_first_all = []
perimeter_improved_all = []
areas = []
perimeter_calculated = []


contour_points_list = []
boxed_in_points = []
possible_contour_points = []

#%%#Start of method definitions for use in "Improved" contour algorithm

def search_eligible_neighbour_layers(point,check_level):
    i = point[0]
    j = point [1]
    eligible_point_list = []
    for check_point in possible_contour_points:
        i_check = check_point[0]
        j_check = check_point[1]
        if((abs(i-i_check)+(abs(j-j_check))<= check_level) and (check_point not in contour_points_list and check_point not in boxed_in_points)) :
            eligible_point_list.append((i_check,j_check))

    return eligible_point_list

def find_neighbour_values(image, i, j, rows, columns):
    if (i > 0 and j < columns - 1 and j > 0):
        top_neighbours_value = (img_2d_2colors[i - 1, j] + img_2d_2colors[i - 1, j - 1] + img_2d_2colors[
            i - 1, j + 1]) / 3
    elif (i > 0 and j < columns - 1):
        top_neighbours_value = (img_2d_2colors[i - 1, j] + img_2d_2colors[i - 1, j] + img_2d_2colors[
            i - 1, j + 1]) / 3
    elif (i > 0 and j > 0):
        top_neighbours_value = (img_2d_2colors[i - 1, j] + img_2d_2colors[i - 1, j - 1] + img_2d_2colors[
            i - 1, j]) / 3
    else:
        top_neighbours_value = 0.0
    if (i < rows - 1 and j < columns - 1 and j > 0):
        down_neighbours_value = (img_2d_2colors[i + 1, j] + img_2d_2colors[i + 1, j - 1] + img_2d_2colors[
            i + 1, j + 1]) / 3
    elif (i < rows - 1 and j < columns - 1):
        down_neighbours_value = (img_2d_2colors[i + 1, j] + img_2d_2colors[i + 1, j] + img_2d_2colors[i + 1, j + 1]) / 3
    elif (i < rows - 1 and j > 0):
        down_neighbours_value = (img_2d_2colors[i + 1, j] +  img_2d_2colors[i + 1, j - 1] + img_2d_2colors[i + 1, j]) / 3
    else:
        down_neighbours_value = 0.0
    if (j > 0 and i < rows - 1 and i > 0):
        left_neighbours_value = (img_2d_2colors[i + 1, j - 1] + img_2d_2colors[i, j - 1] + img_2d_2colors[
            i - 1, j - 1]) / 3
    elif (j > 0 and i < rows - 1):
        left_neighbours_value = (img_2d_2colors[i + 1, j - 1] + img_2d_2colors[i, j - 1] + img_2d_2colors[i, j - 1]) / 3
    elif (j > 0 and i > 0):
        left_neighbours_value = (img_2d_2colors[i, j - 1] + img_2d_2colors[i, j - 1] + img_2d_2colors[i - 1, j - 1]) / 3
    else:
        left_neighbours_value = 0.0
    if (j < columns - 1 and i < rows - 1 and i > 0):
        right_neighbours_value = (img_2d_2colors[i + 1, j + 1] + img_2d_2colors[i, j + 1] + img_2d_2colors[
            i - 1, j + 1]) / 3
    elif (j < columns - 1 and i < rows - 1):
        right_neighbours_value = (img_2d_2colors[i + 1, j + 1] + img_2d_2colors[i, j + 1] + img_2d_2colors[
            i, j + 1]) / 3
    elif (j < columns - 1 and i > 0):
        right_neighbours_value = (img_2d_2colors[i, j + 1] + img_2d_2colors[i, j + 1] + img_2d_2colors[
            i - 1, j + 1]) / 3
    else:
        right_neighbours_value = 0.0

    #print ("NEIGHBOUR AVERAGE VALUES::")
    #print ("top_neighbours_value:", top_neighbours_value)
    #print ("down_neighbours_value:", down_neighbours_value)
    #print ("left_neighbours_value:", left_neighbours_value)
    #print ("right_neighbours_value:", right_neighbours_value)

    ## Sample NEIGHBOUR AVERAGE VALUES::
    # top_neighbours_value: 0.666666666667 - 2/3 on top are white
    # left_neighbours_value: 0.0- all left 3 are black
    # right_neighbours_value: 1.0 - all right 3 are white
    # down_neighbours_value: 0.666666666667 - 2/3 on bottom are white
    # can choose to go left-up or left-down
    ### array indexes for neighbours -
    # 0-top, 1-bottom, 2-left, 3-right
    # 4-TL,5-TR,6-BL,7-BR

    neighbour_values = [top_neighbours_value, down_neighbours_value, left_neighbours_value, right_neighbours_value]
    neighbour_values.append ((neighbour_values[0] + neighbour_values[2]) / 2)
    neighbour_values.append ((neighbour_values[0] + neighbour_values[3]) / 2)
    neighbour_values.append ((neighbour_values[1] + neighbour_values[2]) / 2)
    neighbour_values.append ((neighbour_values[1] + neighbour_values[3]) / 2)

    return neighbour_values


def direction_chooser(direction_list, current_point, points_list):
    points_list_size = len (points_list)
    for direction in direction_list:
        if (direction == 0 and current_point[0] > 0):
            # We are moving top direction
            current_point_temp = (current_point[0] - 1, current_point[1])
            if (current_point_temp not in points_list and current_point_temp not in boxed_in_points):
                current_point = current_point_temp
                #print ("New current_point:", current_point)
                contour_points_list.append (current_point)
                if current_point in possible_contour_points:
                    possible_contour_points.remove (current_point)


                break
        elif (direction == 1 and current_point[0] < rows - 1):
            # We are moving bottom direction
            current_point_temp = (current_point[0] + 1, current_point[1])
            if (current_point_temp not in points_list and current_point_temp not in boxed_in_points):
                current_point = current_point_temp
                #print ("New current_point:", current_point)
                contour_points_list.append (current_point)

                break
        elif (direction == 2 and current_point[1] > 0):
            # We are moving left direction
            current_point_temp = (current_point[0], current_point[1] - 1)
            if (current_point_temp not in points_list and current_point_temp not in boxed_in_points):
                current_point = current_point_temp
                #print ("New current_point:", current_point)
                contour_points_list.append (current_point)
                if current_point in possible_contour_points:
                    possible_contour_points.remove (current_point)

                break
        elif (direction == 3 and current_point[1] < columns - 1):
            # We are moving right direction
            current_point_temp = (current_point[0], current_point[1] + 1)
            if (current_point_temp not in points_list and current_point_temp not in boxed_in_points):
                current_point = current_point_temp
                #print ("New current_point:", current_point)
                contour_points_list.append (current_point)
                if current_point in possible_contour_points:
                    possible_contour_points.remove (current_point)

                break
        elif (direction == 4 and current_point[0] > 0 and current_point[1] > 0):
            # We are moving top-left
            current_point_temp = (current_point[0] - 1, current_point[1] - 1)
            if (current_point_temp not in points_list and current_point_temp not in boxed_in_points):
                current_point = current_point_temp
                #print ("New current_point:", current_point)
                contour_points_list.append (current_point)
                if current_point in possible_contour_points:
                    possible_contour_points.remove (current_point)

                break
        elif (direction == 5 and current_point[0] > 0 and current_point[1] < columns - 1):
            # We are moving top-right
            current_point_temp = (current_point[0] - 1, current_point[1] + 1)
            if (current_point_temp not in points_list and current_point_temp not in boxed_in_points):
                current_point = current_point_temp
                #print ("New current_point:", current_point)
                contour_points_list.append (current_point)

                break
        elif (direction == 6 and current_point[0] < rows - 1 and current_point[1] > 0):
            # We are moving bottom-left
            current_point_temp = (current_point[0] + 1, current_point[1] - 1)
            if (current_point_temp not in points_list and current_point_temp not in boxed_in_points):
                current_point = current_point_temp
                #print ("New current_point:", current_point)
                contour_points_list.append (current_point)
                if current_point in possible_contour_points:
                    possible_contour_points.remove (current_point)
                break
        elif (direction == 7 and current_point[0] < rows - 1 and current_point[1] < columns - 1):
            # We are moving bottom-right
            current_point_temp = (current_point[0] + 1, current_point[1] + 1)
            if (current_point_temp not in points_list and current_point_temp not in boxed_in_points):
                current_point = current_point_temp
                #print ("New current_point:", current_point)
                contour_points_list.append (current_point)
                if current_point in possible_contour_points:
                    possible_contour_points.remove (current_point)

                break

    return direction, contour_points_list, current_point


def current_point_check_adjust_assign(point_to_check, img_2d_2colors, rows, columns, adjust_boolean, fallback_boolean= False, change_level=1):
    if ((point_to_check in boxed_in_points and adjust_boolean) or fallback_boolean):
        i = point_to_check[0]
        j = point_to_check[1]

        neighbour_values = find_neighbour_values (img_2d_2colors, point_to_check[0], point_to_check[1],
                                                  rows, columns)
        average_neighbour_value = np.average (neighbour_values)

        # if (average_neighbour_value > 0.5):
        #     print ("ENTERING FALLBACK CORRECTION-majority white around- go in")
        #     if (change_level < i < ((rows - 1) / 2)):
        #         i = i + change_level
        #     elif (i > ((rows - 1) / 2) and i < (rows - change_level)):
        #         i = i - change_level
        #     if (change_level < j and j < (columns - 1) / 2):
        #         j = j + change_level
        #     elif ((columns - 1) / 2 < j and j < (columns - change_level)):
        #         j = j - change_level
        # if (average_neighbour_value <= 0.5):
        #     print ("ENTERING FALLBACK CORRECTION-surrounded black- go out")
        #     if (change_level < i < (rows - 1) / 2):
        #         i = i - change_level
        #     elif ((rows - 1) / 2 < i < (rows - change_level)):
        #         i = i + change_level
        #     if (change_level < j < (columns - 1) / 2):
        #         j = j - change_level
        #     elif ((columns - 1) / 2 < j < (columns - change_level)):
        #         j = j + change_level

        current_point = (i,j)

        min_level = 35
        if((current_point in possible_contour_points and current_point not in contour_points_list and current_point not in boxed_in_points)):
            print("New fallback point is ok")

        elif(change_level > min_level and change_level <= min_level+10):
             print("NO ELIGIBLE POINTS UPTO 40 levels of neighbours")
             actual_change = change_level - min_level
             if (average_neighbour_value > 0.5):
                 #print ("ENTERING FALLBACK CORRECTION-majority white around- go in")
                 if (change_level < i < ((rows - 1) / 2)):
                     i = i + change_level
                 elif (i > ((rows - 1) / 2) and i < (rows - change_level)):
                     i = i - change_level
                 if (change_level < j and j < (columns - 1) / 2):
                     j = j + change_level
                 elif ((columns - 1) / 2 < j and j < (columns - change_level)):
                     j = j - change_level
             if (average_neighbour_value <= 0.5):
                 #print ("ENTERING FALLBACK CORRECTION-surrounded black- go out")
                 if (change_level < i < (rows - 1) / 2):
                     i = i - change_level
                 elif ((rows - 1) / 2 < i < (rows - change_level)):
                     i = i + change_level
                 if (change_level < j < (columns - 1) / 2):
                     j = j - change_level
                 elif ((columns - 1) / 2 < j < (columns - change_level)):
                     j = j + change_level



             current_point = (i, j)
             if(current_point not in contour_points_list and current_point not in boxed_in_points):
                 print("New Extreme fallback point ok")
             else:
                 current_point = current_point_check_adjust_assign (current_point,
                                                                    img_2d_2colors, rows,
                                                                    columns, True, True, change_level + 1)

        elif (change_level > min_level+10):
            #print("I GIVE UP")
            for random_point in possible_contour_points:
                if(random_point not in contour_points_list and random_point not in boxed_in_points and random_point !=starting_point):
                    current_point = random_point
                    break
        else:
            el_point_list = search_eligible_neighbour_layers (current_point, change_level)
            if (len (el_point_list) > 0):
                current_point = el_point_list[0]
            else:
                current_point = current_point_check_adjust_assign (current_point,
                                                                   img_2d_2colors, rows,
                                                                   columns, True, True, change_level + 1)



        return current_point
    else:
        return point_to_check

#End of method definitions for use in "Improved" contour algorithm
############################################################################
#%%Main Start
seed = 42

cluster_count = 3

file_name_list_lr = ["low_risk_" + str (x) for x in range (1, 12)]
file_name_list_mr = ["medium_risk_" + str (x) for x in range (1, 17)]
file_name_list_melanoma = ["melanoma_" + str (x) for x in range (1, 28)]

file_name_list = file_name_list_lr + file_name_list_mr +file_name_list_melanoma


for file_name in file_name_list:
    print ("File being processed:", file_name)

    # %% reading file and shaping it and then applying Kmeans fit
    file_path = 'Datasource/' + file_name + '.jpg'

    im = mpimg.imread (file_path)

    [N1, N2, N3] = im.shape
    im_2D = im.reshape ((N1 * N2, N3))

    kmeans = KMeans (n_clusters=3, random_state=seed)
    kmeans.fit (im_2D)

    centroids = kmeans.cluster_centers_.astype ('uint8')
    labels = kmeans.labels_

    im_2D_3Colors = kmeans.predict (im_2D).reshape ((N1, N2))

    # %% Finding the index of the centroid with the darkest color and plotting the
    # new image with just three colors
    darkest_color_index = np.unique (centroids.argmin (axis=0))[0]
    darkest_color = centroids[darkest_color_index]

    plt.figure ( )
    plt.imshow (im)

    plt.figure ( )
    plt.imshow (im_2D_3Colors)
    plt.savefig ("Output/Kmeans/" + file_name + "_kmeans.png")

    # %%Finding median index
    darkest_pixels_check = (labels == darkest_color_index)
    darkest_pixels_indexes = np.where (darkest_pixels_check == True)
    darkest_pixels_indexes = np.array (darkest_pixels_indexes)
    darkest_median_pixel_index = np.int (np.median (darkest_pixels_indexes, axis=1))
    #print ("darkest_color_index:", darkest_color_index)

    # %% Starting from the median to find the borders
    darkest_median_pixel_ROW_INDEX = darkest_median_pixel_index // N2
    darkest_median_pixel_Column_INDEX = darkest_median_pixel_index % N2

    print ("darkest_median_pixel_ROW_INDEX:", darkest_median_pixel_ROW_INDEX)
    print ("darkest_median_pixel_Column_INDEX:", darkest_median_pixel_Column_INDEX)

    # print("im_2D_3Colors[:, i]-231=42:",im_2D_3Colors[:,231])

    top_border = 0
    bottom_border = 0
    right_border = 0
    left_border = 0
    j = 0
    # Use np non zero count

    for i in range (darkest_median_pixel_ROW_INDEX, 0, -1):
        darkest_pixels_in_row = np.count_nonzero (im_2D_3Colors[i, :] == darkest_color_index)

        # check to see if at least 2 pixels match darkest color in current row
        if (darkest_pixels_in_row > 1):
            top_border = i
            j = 0
        # If not, check to see if it's the case for FIVE more columns to avoid outliers
        elif (j > 5):
            print ("border reached on the top at:", top_border)
            break
        else:
            j = j + 1

    j = 0
    for i in range (darkest_median_pixel_ROW_INDEX, N1, 1):
        darkest_pixels_in_row = np.count_nonzero (im_2D_3Colors[i, :] == darkest_color_index)
        # check to see if at least 2 pixels match darkest color in current row
        if (darkest_pixels_in_row > 1):
            j = 0
            bottom_border = i
        # If not, check to see if it's the case for FIVE more columns to avoid outliers
        elif (j > 5):

            print ("border reached on the bottom at:", bottom_border)
            break
        else:
            j = j + 1

    for i in range (darkest_median_pixel_Column_INDEX, 0, -1):

        darkest_pixels_in_column = np.count_nonzero (im_2D_3Colors[:, i] == darkest_color_index)
        # print ("darkest_pixels_in_column -20:",i-20, np.count_nonzero (im_2D_3Colors[:, i-20] == darkest_color_index))
        # check to see if at least 2 pixels match darkest color in current column
        if (darkest_pixels_in_column > 1):
            left_border = i
            j = 0
        # If not, check to see if it's the case for five more rows to avoid outliers
        elif (j > 5):
            print ("border reached on the left at:", left_border)
            break
        else:
            j = j + 1

    for i in range (darkest_median_pixel_Column_INDEX, N2, 1):
        darkest_pixels_in_column = np.count_nonzero (im_2D_3Colors[:, i] == darkest_color_index)
        # check to see if at least 2 pixels match darkest color in current column
        if (darkest_pixels_in_column > 1):
            j = 0
            right_border = i
        # If not, check to see if it's the case for FIVE more rows to avoid outliers
        elif (j > 5):
            print ("border reached on the right at:", right_border)
            break
        else:
            j = j + 1

    # %% Cropping the image
    img_2d_3colors_cpy = im_2D_3Colors
    # add margins to borders
    if (top_border > 10): top_border = top_border - 10
    if (bottom_border < N1 - 10): bottom_border = bottom_border + 10
    if (left_border > 10): left_border = left_border - 10
    if (right_border < N2 - 10): right_border = right_border + 10
    img_2d_3colors_cpy = img_2d_3colors_cpy[top_border:bottom_border, left_border:right_border]
    plt.figure ( )
    plt.imshow (img_2d_3colors_cpy)
    plt.savefig ("Output/cropped/" + file_name + "_kmeans_crop.png")

    # np.savetxt ("Output/" + file_name + "img_2d_3colors_cpy.csv", img_2d_3colors_cpy, delimiter=',')

    # %%Change 3 colours to 2 colours
    img_2d_2colors = img_2d_3colors_cpy
    img_2d_2colors[img_2d_2colors != darkest_color_index] = 4
    img_2d_2colors[img_2d_2colors == darkest_color_index] = 0
    darkest_color_index_new = 0
    img_2d_2colors[img_2d_2colors != darkest_color_index_new] = 1

    plt.figure ( )
    plt.imshow (img_2d_2colors)
    plt.savefig ("Output/2Colors/" + file_name + "_2Colors_crop.png")
    # np.savetxt ("Output/" + file_name + "img_2d_2colors_cpy.csv", img_2d_2colors, delimiter=',')

    # %% Smoothening to remove small outliers
    ## if majority pixels surrounding a light pixel are dark, make it dark

    [rows, columns] = img_2d_2colors.shape

    for i in range (1, rows - 1):
        for j in range (1, columns - 1):
            if (img_2d_2colors[i, j] != darkest_color_index_new):
                average_neighbour_value = (img_2d_2colors[i - 1, j] + img_2d_2colors[i - 1, j - 1] + img_2d_2colors[
                    i - 1, j + 1] + \
                                           img_2d_2colors[i + 1, j] + img_2d_2colors[i + 1, j - 1] + img_2d_2colors[
                                               i + 1, j - 1] + \
                                           img_2d_2colors[i, j - 1] + img_2d_2colors[i, j + 1]) / 8
                if (average_neighbour_value < (4 / 8)):
                    img_2d_2colors[i, j] = darkest_color_index_new

    # %% Contour algorithm -Original/First
    dark_region_contour_pixels_rowwise = []
    dark_region_contour_pixels_colwise = []
    in_dark_region_flag = 0
    region_counter = 0

    # Use flags to find contours and use counters to avoid small outliers
    for i in range (rows):
        region_counter = 0
        for j in range (columns):

            if (img_2d_2colors[i, j] == darkest_color_index_new and in_dark_region_flag == 0
                    and (j < 6 or region_counter > 0)):

                dark_region_contour_pixels_rowwise.append ((i, j))
                in_dark_region_flag = 1
                in_light_region_flag = 0
                region_counter = 0
            elif (img_2d_2colors[i, j] != darkest_color_index_new and in_dark_region_flag == 1
                  and (j < 6 or region_counter > 0)):

                dark_region_contour_pixels_rowwise.append ((i, j - 1))
                in_dark_region_flag = 0
                in_light_region_flag = 1
                region_counter = 0
            else:
                region_counter = region_counter + 1

    for j in range (columns):
        region_counter = 0
        for i in range (rows):
            if (img_2d_2colors[i, j] == darkest_color_index_new and in_dark_region_flag == 0
                    and (i < 6 or region_counter > 0)):
                dark_region_contour_pixels_colwise.append ((i, j))
                in_dark_region_flag = 1
                in_light_region_flag = 0
                region_counter = 0
            elif (img_2d_2colors[i, j] != darkest_color_index_new and in_dark_region_flag == 1
                  and (i < 6 or region_counter > 0)):
                dark_region_contour_pixels_colwise.append ((i - 1, j))
                in_dark_region_flag = 0
                in_light_region_flag = 1
                region_counter = 0
            else:
                region_counter = region_counter + 1

    contour_pixels_line = np.zeros ((rows, columns))

    # print("dark_region_contour_pixels_rowwise:",dark_region_contour_pixels_rowwise)

    contour_pixels = dark_region_contour_pixels_rowwise + dark_region_contour_pixels_colwise

    for i in contour_pixels:
        contour_pixels_line[i] = 1

    plt.matshow (contour_pixels_line, cmap="Blues")
    plt.savefig ("Output/Contour/" + file_name + "_contour.png")

    ###################################################################
    # %% Improved/ New Contour methodology
    [rows, columns] = img_2d_2colors.shape
    print ("rows:", rows)
    print ("columns:", columns)
    darkest_pixels_check_2D = (img_2d_2colors == 0)
    darkest_pixels_indexes_2D = np.where (darkest_pixels_check_2D == True)
    darkest_pixels_indexes_2D = np.array (darkest_pixels_indexes_2D)
    print ( )
    darkest_median_pixel_index_2D = (np.median (darkest_pixels_indexes_2D, axis=1))

    # %% starting from the median to start
    darkest_median_pixel_ROW_INDEX_2D = np.int (darkest_median_pixel_index_2D[0])
    darkest_median_pixel_Column_INDEX_2D = np.int (darkest_median_pixel_index_2D[1])

    print ("darkest_median_pixel_ROW_INDEX_2D:", darkest_median_pixel_ROW_INDEX_2D)
    print ("darkest_median_pixel_Column_INDEX_2D:", darkest_median_pixel_Column_INDEX_2D)

    in_dark_region_flag = 1
    dark_pixel_counter = 0

    # Create a list of points to be avoided by the improved contour algorithm
    possible_contour_points = []
    boxed_in_points = []
    for j in range (columns):
        for i in range (rows):
            neighbour_values_check = find_neighbour_values (img_2d_2colors, i, j, rows, columns)
            if (np.median (neighbour_values_check) == 0.0 or np.median (neighbour_values_check) == 1.0):
                boxed_in_points.append ((i, j))
            else:
                possible_contour_points.append((i,j))

    #print ("boxed_in_points:", boxed_in_points)

    possible_contour_points_line = np.zeros((rows,columns))
    for i in possible_contour_points:
        possible_contour_points_line[i] = 1

    plt.matshow (possible_contour_points_line, cmap="Blues")
    plt.savefig ("Output/Contour/" + file_name + "_POSSIBILITY.png")


    # Finding the starting point - for new contour search alogirthm
    current_row = darkest_median_pixel_ROW_INDEX_2D
    counter = 0

    right_border_contour  = darkest_median_pixel_Column_INDEX_2D
    img_2d_2colors_copy_for_starting_point = np.array(img_2d_2colors)
    for column_value in range (darkest_median_pixel_Column_INDEX_2D, columns, 1):
        darkest_pixels_in_column = np.count_nonzero(img_2d_2colors_copy_for_starting_point[:, column_value] == 0)
        # check to see if at least 2 pixels match darkest color in current column
        if (darkest_pixels_in_column > 1):
            counter = 0
            right_border_contour = column_value
        # If not, check to see if it's the case for two more rows to avoid outliers
        elif (counter > 2):
            print ("border reached on the right at:", right_border_contour)
            break
        else:
            j = j + 1
    #%% Fallback algorithm to find starting point for main contour algorithm

    #Only if the above algorithm to find starting point fails, a fallback
    if(right_border_contour == darkest_median_pixel_Column_INDEX_2D or right_border_contour < columns/2):
        for current_column in range (darkest_median_pixel_Column_INDEX_2D, columns, 1):
            if (img_2d_2colors[current_row, current_column] == darkest_color_index_new and in_dark_region_flag == 1):
                dark_pixel_counter = dark_pixel_counter + 1
                counter = 0

            elif (img_2d_2colors[current_row, current_column] == darkest_color_index_new and in_dark_region_flag == 0):
                    dark_pixel_counter = dark_pixel_counter + 1
                    in_dark_region_flag = 1
                    counter = 0

                ##Nothing to do
            elif (img_2d_2colors[current_row, current_column] != darkest_color_index_new and in_dark_region_flag == 1):
                right_border_contour = current_column - 1
                in_dark_region_flag = 0
                if(counter == 0):
                 counter +=1
                 counting_border_contour= current_column-1
                else:
                 counter = 1
                 counting_border_contour = current_column - 1
            elif (img_2d_2colors[current_row, current_column] != darkest_color_index_new and in_dark_region_flag == 0 and counter > 0):
                counter += 1
                #If the border from dark to light color happened and no color change happened ,
                # we assume the color change column at this row to be part of the contour
                if(counter > 2):
                    right_border_contour = current_column-3

                    break

        if(counter<2):
            right_border_contour= counting_border_contour

    starting_point = (current_row, right_border_contour)

    current_point = starting_point

    current_point_prev = (0, 0)
    go_back_counter = 0
    go_back_main_counter = 0
    go_back_check_point = (0, 0)
    prev_10_directions_list = np.zeros (10)
    prev_10_points_list = [current_point]
    current_point_main_loop_check = (0,0)
    current_point_main_loop_check_count = 0
    # Counter for How many times to loop to find the entire contour
    contour_points_list = []
    for k in range (1, 5500):
        i = current_point[0]
        j = current_point[1]

        print ("i:", i)
        print ("j:", j)

        if(current_point_main_loop_check == current_point):
            current_point_main_loop_check_count +=1
        else:
            current_point_main_loop_check_count = 0

        print ("current_point_main_loop_check_count:",current_point_main_loop_check_count)
        if(current_point_main_loop_check_count > 10):
            boxed_in_points.append(current_point)
            if current_point in possible_contour_points:
                possible_contour_points.remove(current_point)
            current_point = current_point_check_adjust_assign(current_point, img_2d_2colors, rows, columns,
                                                                   True,True)
            current_point_main_loop_check_count = 0

        current_point_main_loop_check = current_point

        print ("current_point:", current_point)


        if (i < rows and j < columns):

            neighbour_values = find_neighbour_values (img_2d_2colors, i, j, rows, columns)

            # masked array to find min value
            # masked_NV = np.ma.masked_equal (neighbour_values, 0.0, copy=False)
            # masked_NV = np.ma.masked_equal (masked_NV, 1.0, copy=False)
            # print("np.ma.median(masked_NV):",np.ma.median(masked_NV))
            if (np.median (neighbour_values) == 0.0 or np.median (
                    neighbour_values) == 1.0 or current_point == current_point_prev):
                print ("Boxed INNNN - Going back !-currentpont-", current_point,"::current_point_prev:",current_point_prev)
                print("go_back_counter:",go_back_counter)
                print ("go_back_main_counter:", go_back_main_counter)
                # the pixel is surrounded by all white or all back, better to go back adn choose the second best option and see
                if (current_point == current_point_prev):
                    go_back_counter = go_back_counter + 1


                if (current_point in contour_points_list):
                    contour_points_list.remove (current_point)
                    possible_contour_points.append (current_point)

                # if (current_point_prev not in boxed_in_points):
                # False for adjust, as we want the go  back part of the algorithm to be applied
                current_point = current_point_check_adjust_assign (current_point_prev, img_2d_2colors, rows, columns,
                                                                   False)

                print ("Going back to !-current_pointPREV-", current_point)
                if (go_back_check_point == current_point):
                    go_back_counter = go_back_counter + 1
                else:
                    go_back_check_point = current_point
                    go_back_counter = 0
                    go_back_main_counter = 0

                neighbour_values = find_neighbour_values (img_2d_2colors, current_point[0], current_point[1], rows,
                                                          columns)
                average_neighbour_value = np.average (neighbour_values)
                sorted_neighbour_Values = np.array (
                    sorted (neighbour_values, key=lambda x: abs (x - np.median (neighbour_values))))
                # Removing the possible previously chosen value for prev neighbour
                # sorted_neighbour_Values_cpy = sorted_neighbour_Values

                for gbc in range (go_back_counter + 1):

                    if (sorted_neighbour_Values.size > 0):
                        value = sorted_neighbour_Values[0]

                        sorted_neighbour_Values = np.extract (sorted_neighbour_Values != value, sorted_neighbour_Values)
                    else:
                        # print("No more neighbour values  available- This  point has no route- go back once more")
                        go_back_main_counter += 1

                # sorted_neighbour_Values not available, go back one more level
                if (sorted_neighbour_Values.size == 0 and go_back_main_counter < len(prev_10_points_list)):
                    prev_10_points_list = prev_10_points_list[:-go_back_main_counter or None]
                    #contour_points_list = contour_points_list[:-go_back_main_counter or None]
                    boxed_in_points.append(current_point)
                    if(current_point in possible_contour_points):
                        possible_contour_points.remove (current_point)
                    if (current_point in contour_points_list):
                        contour_points_list.remove (current_point)

                    if (current_point in prev_10_points_list):
                        prev_10_points_list.remove (current_point)

                    # 10 points -highest index - 9, we checked highest index, so go further- 8

                    prev_point_index = len(prev_10_points_list) - go_back_main_counter

                    if (prev_point_index > (len (prev_10_points_list) - 1) and len (prev_10_points_list) > 0):
                        prev_point_index = (len (prev_10_points_list) - 1)
                    elif (len (prev_10_points_list) <= 0):
                        print ("not enough previous points history available")
                        current_point = current_point_check_adjust_assign (current_point,
                                                                       img_2d_2colors, rows,
                                                                       columns, True, True)


                    print ("prev_point_index to use:", prev_point_index)

                    # False for adjust flag so the go back counting works
                    if(len (prev_10_points_list) > 0):
                        current_point = current_point_check_adjust_assign (prev_10_points_list[prev_point_index],
                                                                       img_2d_2colors, rows,
                                                                       columns, False)
                    else:
                        current_point = current_point_check_adjust_assign (current_point,
                                                                       img_2d_2colors, rows,
                                                                       columns, True,True)
                    print("current_point after going back 1 using prev point list::::", current_point)

                    neighbour_values = find_neighbour_values (img_2d_2colors, current_point[0], current_point[1], rows,
                                                              columns)
                    sorted_neighbour_Values = np.array (
                        sorted (neighbour_values, key=lambda x: abs (x - np.median (neighbour_values))))

                    # check if all 10 points n history checked or if history is just repeating between few points
                    # if so, try to get out of loop
                elif (go_back_main_counter >= 10 ):
                    print ("checked back 10 points in history-everything failed")
                    if (current_point in contour_points_list):
                        contour_points_list.remove (current_point)
                    if (current_point in prev_10_points_list):
                        prev_10_points_list.remove (current_point)
                    current_point_prev_check_Test = current_point
                    fall_back_direction = np.median (prev_10_directions_list)
                    direction, contour_points_list, current_point = direction_chooser ([fall_back_direction],
                                                                                       current_point,
                                                                                       contour_points_list)
                    # If median direction approach fails - go far final fallback
                    if (current_point_prev_check_Test == current_point):
                        current_point = current_point_check_adjust_assign (current_point,
                                                                           img_2d_2colors, rows,
                                                                           columns, True, True)

                    print ("current_point. MIDDLE FALLBACK CORRECTION:", current_point)

                    if(current_point_prev_check_Test== current_point):
                        boxed_in_points.append(current_point_prev_check_Test)
                        if(current_point_prev_check_Test in possible_contour_points):
                            possible_contour_points.remove (current_point_prev_check_Test)

                        current_point = current_point_check_adjust_assign (current_point,
                                                                           img_2d_2colors, rows,
                                                                           columns, True, True)


                    print ("current_point. AFTER FALLBACK CORRECTION:", current_point)





                else:
                    go_back_main_counter = 0


            else:
                go_back_counter = 0
                go_back_main_counter = 0
                sorted_neighbour_Values = np.array (
                    sorted (neighbour_values, key=lambda x: abs (x - np.median (neighbour_values))))

            neighbour_values = find_neighbour_values (img_2d_2colors, current_point[0], current_point[1], rows, columns)
            sorted_neighbour_Values = np.array (sorted (neighbour_values, key=lambda x: abs (
                x - np.median (neighbour_values))))

            print ("sorted_neighbour_Values:", sorted_neighbour_Values)
            # sorted(a, key=lambda x: abs (x - np.ma.median(a))))

            current_point_prev = current_point

            fall_back_direction = np.median (prev_10_directions_list)
            ### array indexes for neighbours -
            # 0-top, 1-bottom, 2-left, 3-right
            # 4-TL,5-TR,6-BL,7-BR
            # Anticlockwise order ==> 0,4,2,6,1,7,3,5
            list (filter (lambda x: x != 1.0 and x != 0.0, sorted_neighbour_Values))
            for sorted_neighbour_Value in sorted_neighbour_Values:

                direction_list_Value = np.where (neighbour_values == sorted_neighbour_Value)
                #print ("direction_chosen-k:", direction_list_Value,"::::",k)

                directions_chosen_list = []
                if (len (direction_list_Value) > 0):

                    if (0 in direction_list_Value[0]):
                        directions_chosen_list.append(0)
                    if (4 in direction_list_Value[0]):
                        directions_chosen_list.append(4)
                    if (2 in direction_list_Value[0]):
                        directions_chosen_list.append(2)
                    if (6 in direction_list_Value[0]):
                        directions_chosen_list.append(6)
                    if (1 in direction_list_Value[0]):
                        directions_chosen_list.append(1)
                    if (7 in direction_list_Value[0]):
                        directions_chosen_list.append(7)
                    if (3 in direction_list_Value[0]):
                        directions_chosen_list.append(3)
                    if (5 in direction_list_Value[0]):
                        directions_chosen_list.append(5)
                    direction, contour_points_list, current_point = direction_chooser (directions_chosen_list,
                                                                                       current_point,
                                                                                       contour_points_list)
                    if (current_point_prev != current_point):
                        break
            if (current_point == current_point_prev):
                direction, contour_points_list, current_point = direction_chooser ([fall_back_direction], current_point,
                                                                                   contour_points_list)

            #print("contour_points_list:",contour_points_list)
            if (current_point != current_point_prev and current_point not in prev_10_points_list):
                prev_10_directions_list = np.delete (prev_10_directions_list, 0)
                prev_10_directions_list = np.append (prev_10_directions_list, direction)
                print ("prev_10_points_list-old:", prev_10_points_list)
                if (len (prev_10_points_list) >= 10):
                    prev_10_points_list.pop (0)
                prev_10_points_list.append (current_point)
                print ("prev_10_points_list-new:", prev_10_points_list)
            print("Prev 10 points list:::", prev_10_points_list)

            if ((current_point == starting_point and len(contour_points_list) > 75) or len(possible_contour_points) <= 0 or go_back_main_counter > 100):
                print("Reached starting point- exiting algorithm")
                contour_points_list.append(starting_point)
                break


    contour_pixels_line_improved = np.zeros ((rows, columns))


    for i in contour_points_list:
        contour_pixels_line_improved[i] = 1


    plt.matshow (contour_pixels_line_improved, cmap="Blues")
    plt.savefig ("Output/Contour/" + file_name + "_contour_IMPROVED.png")


    # %% area and perimeter
    area_dark = np.count_nonzero (img_2d_2colors == 0)
    perimeter_contour = np.count_nonzero (contour_pixels_line == 1)
    perimeter_Contour_Improved = np.count_nonzero (contour_pixels_line_improved == 1)

    radius_circle = np.sqrt (area_dark / np.pi)
    perimeter_circle = 2 * np.pi * radius_circle

    ratio_contour = perimeter_contour / perimeter_circle
    ratio_contour_improved = perimeter_Contour_Improved/ perimeter_circle


    ratio_first_all.append(ratio_contour)
    ratio_improved_all.append(ratio_contour_improved)
    perimeter_first_all.append(perimeter_contour)
    perimeter_improved_all.append(perimeter_Contour_Improved)
    areas.append(area_dark)
    perimeter_calculated.append(perimeter_circle)

    if "low_risk" in file_name:
        ratio_lr_first.append (ratio_contour)
        ratio_lr_improved.append(ratio_contour_improved)

    elif "medium_risk" in file_name:
        ratio_mr_first.append (ratio_contour)
        ratio_mr_improved.append(ratio_contour_improved)

    else:
        ratio_hr_first.append (ratio_contour)
        ratio_hr_improved.append(ratio_contour_improved)

    plt.close ("all")



    print ("Basta")

print ("Low risk_first:", np.mean (ratio_lr_first))
print ("Medium risk_first:", np.mean (ratio_mr_first))
print ("Melanoma_first:", np.mean (ratio_hr_first))

print ("Low risk_IMPROVED:", np.mean (ratio_lr_improved))
print ("Medium risk_IMPROVED:", np.mean (ratio_mr_improved))
print ("Melanoma_risk_IMPROVED:", np.mean (ratio_hr_improved))

# %% plot
# import matplotlib.pyplot as plt
# ratio_lr_first = [0,1]
# ratio_mr_first = [2,3]
# ratio_hr_first = [4,5]
#
# ratio_lr_improved = [1,1]
# ratio_mr_improved = [2,2]
# ratio_hr_improved = [3,3]


plt.figure ( )
lr = plt.plot (ratio_lr_first, color='red', label='Low Risk')
mr = plt.plot (ratio_mr_first, color='blue', label='Medium Risk')
hr = plt.plot (ratio_hr_first, color='green', label='Melanoma')
plt.xlabel ("Picture")
plt.ylabel ("Ratio")
plt.title ("Ratio between perimeter of the mole and perimeter of the circle with the same area")
plt.grid ( )
plt.legend ( )

#plt.savefig ("Output/Final/" + file_name + "_ratios.png")
plt.savefig ("Output/Final/" + "a" + "_ratios.png")


plt.figure ( )
lr_improved = plt.plot (ratio_lr_improved, color='red', label='Low Risk')
mr_improved = plt.plot (ratio_mr_improved, color='blue', label='Medium Risk')
hr_improved = plt.plot (ratio_hr_improved, color='green', label='Melanoma')
plt.xlabel ("Picture")
plt.ylabel ("Ratio")
plt.title ("IMPROVED -- Ratio between perimeter of the mole and perimeter of the circle with the same area")
plt.grid ( )
plt.legend ( )
plt.savefig ("Output/Final/" + "b" + "_ratios_IMPROVED.png")

plt.close ("all")


