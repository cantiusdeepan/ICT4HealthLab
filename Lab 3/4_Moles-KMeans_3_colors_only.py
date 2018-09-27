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

cluster_count = 3

file_name_list_lr = ["low_risk_"+str(x) for x in range (1, 12)]
file_name_list_mr = ["medium_risk_"+str(x) for x in range (1, 17)]
file_name_list_melanoma = ["melanoma_"+str(x) for x in range (1, 28)]

file_name_list = file_name_list_lr + file_name_list_mr+file_name_list_melanoma



for file_name in file_name_list:
    print ("File being processed:",file_name)

    # %% reading file and shaping it and then applying Kmeans fit
    file_path = 'Datasource/' + file_name + '.jpg'

    im = mpimg.imread (file_path)

    [N1, N2, N3] = im.shape
    im_2D = im.reshape ((N1 * N2, N3))



    kmeans = KMeans (n_clusters=3, random_state=seed)
    kmeans.fit (im_2D)

    centroids = kmeans.cluster_centers_.astype('uint8')
    labels = kmeans.labels_

    im_2D_3Colors = kmeans.predict(im_2D).reshape ((N1, N2))


    # %% finding the index of the centroid with the darkest color and plotting the
    # new image with just three colors
    darkest_color_index = np.unique (centroids.argmin (axis=0))[0]
    darkest_color = centroids[darkest_color_index]
   

    plt.figure ( )
    plt.imshow (im)


    plt.figure ( )
    plt.imshow (im_2D_3Colors)
    plt.savefig ("Output/Kmeans/" + file_name + "_kmeans.png")


    # %%finding median index
    darkest_pixels_check = (labels == darkest_color_index)
    darkest_pixels_indexes = np.where(darkest_pixels_check==True)
    darkest_pixels_indexes = np.array (darkest_pixels_indexes)
    darkest_median_pixel_index = np.int (np.median (darkest_pixels_indexes, axis=1))
    print("darkest_color_index:",darkest_color_index)


    #%% starting from the median to find the borders
    darkest_median_pixel_ROW_INDEX = darkest_median_pixel_index//N2
    darkest_median_pixel_Column_INDEX = darkest_median_pixel_index%N2


    print("darkest_median_pixel_ROW_INDEX:", darkest_median_pixel_ROW_INDEX)
    print("darkest_median_pixel_Column_INDEX:", darkest_median_pixel_Column_INDEX)

    #print("im_2D_3Colors[:, i]-231=42:",im_2D_3Colors[:,231])



    top_border = 0
    bottom_border = 0
    right_border = 0
    left_border = 0
    j=0
    #Use np non zero count

    for i in range(darkest_median_pixel_ROW_INDEX,0,-1):
        darkest_pixels_in_row = np.count_nonzero (im_2D_3Colors[i,:] == darkest_color_index)

        #check to see if at least 2 pixels match darkest color in current row
        if(darkest_pixels_in_row > 1):
            top_border = i
            j=0
        #If not, check to see if it's the case for FIVE more columns to avoid outliers
        elif(j>5):
            print("border reached on the top at:",top_border)
            break
        else:
            j=j+1


    j = 0
    for i in range (darkest_median_pixel_ROW_INDEX, N1, 1):
        darkest_pixels_in_row = np.count_nonzero (im_2D_3Colors[i, :] == darkest_color_index)
        # check to see if at least 2 pixels match darkest color in current row
        if (darkest_pixels_in_row > 1):
            j = 0
            bottom_border = i
        #If not, check to see if it's the case for FIVE more columns to avoid outliers
        elif (j > 5):

            print ("border reached on the bottom at:", bottom_border)
            break
        else:
            j = j + 1


    for i in range (darkest_median_pixel_Column_INDEX, 0, -1):

        darkest_pixels_in_column = np.count_nonzero (im_2D_3Colors[:, i] == darkest_color_index)
        #print ("darkest_pixels_in_column -20:",i-20, np.count_nonzero (im_2D_3Colors[:, i-20] == darkest_color_index))
        # check to see if at least 2 pixels match darkest color in current column
        if (darkest_pixels_in_column> 1):
            left_border = i
            j = 0
        #If not, check to see if it's the case for five more rows to avoid outliers
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


    #%% crop
    img_2d_3colors_cpy = im_2D_3Colors
    #add margins to borders
    if(top_border >10): top_border = top_border-10
    if(bottom_border<N1-10): bottom_border = bottom_border +10
    if(left_border>10): left_border = left_border - 10
    if(right_border < N2 - 10): right_border = right_border + 10
    img_2d_3colors_cpy = img_2d_3colors_cpy[ top_border:bottom_border,left_border:right_border]
    plt.figure ( )
    plt.imshow (img_2d_3colors_cpy)
    plt.savefig ("Output/cropped/" + file_name + "_kmeans_crop.png")

    #np.savetxt ("Output/" + file_name + "img_2d_3colors_cpy.csv", img_2d_3colors_cpy, delimiter=',')

    #%%Change 3 colours to 2 colours
    img_2d_2colors = img_2d_3colors_cpy
    img_2d_2colors[img_2d_2colors != darkest_color_index] = 4
    img_2d_2colors[img_2d_2colors == darkest_color_index] = 0
    darkest_color_index_new = 0
    img_2d_2colors[img_2d_2colors != darkest_color_index_new] = 1

    plt.figure ( )
    plt.imshow (img_2d_2colors)
    plt.savefig ("Output/2Colors/" + file_name + "_2Colors_crop.png")
    #np.savetxt ("Output/" + file_name + "img_2d_2colors_cpy.csv", img_2d_2colors, delimiter=',')

    #%% Smoothening to remove small outliers
    ## if majority pixels surrounding a light pixel are dark, make it dark

    [rows, columns] = img_2d_2colors.shape

    for i in range(1,rows-1):
        for j in range(1,columns-1):
            if (img_2d_2colors[i, j] != darkest_color_index_new ):
                average_neighbour_value =  (img_2d_2colors[i-1, j] + img_2d_2colors[i-1, j-1]+ img_2d_2colors[i-1, j+1] + \
                             img_2d_2colors[i+1, j]+ img_2d_2colors[i+1, j-1]  +img_2d_2colors[i+1, j-1]+\
                             img_2d_2colors[i, j-1] + img_2d_2colors[i, j+1])/8
                if(average_neighbour_value < (4/8)):
                    img_2d_2colors[i, j] = darkest_color_index_new




    #%% Contouring
    dark_region_contour_pixels_rowwise = []
    dark_region_contour_pixels_colwise = []
    in_dark_region_flag = 0
    region_counter = 0


    #Use flags to find contours and use counters to avoid small outliers
    for i in range(rows):
        region_counter = 0
        for j in range(columns):

            if(img_2d_2colors[i,j]==darkest_color_index_new and in_dark_region_flag==0
                    and (j<6 or region_counter > 0)):

                dark_region_contour_pixels_rowwise .append((i,j))
                in_dark_region_flag = 1
                in_light_region_flag = 0
                region_counter = 0
            elif(img_2d_2colors[i,j]!=darkest_color_index_new and in_dark_region_flag==1
                 and (j<6 or region_counter > 0)):

                dark_region_contour_pixels_rowwise.append ((i, j-1))
                in_dark_region_flag = 0
                in_light_region_flag = 1
                region_counter = 0
            else:
                region_counter = region_counter +1


    for j in range(columns):
        region_counter = 0
        for i in range(rows):
            if(img_2d_2colors[i,j]==darkest_color_index_new and in_dark_region_flag==0
                    and (i < 6 or region_counter > 0)):
                dark_region_contour_pixels_colwise.append((i,j))
                in_dark_region_flag = 1
                in_light_region_flag = 0
                region_counter = 0
            elif(img_2d_2colors[i,j]!=darkest_color_index_new and in_dark_region_flag==1
                 and (i < 6 or region_counter > 0)):
                dark_region_contour_pixels_colwise.append ((i-1, j))
                in_dark_region_flag = 0
                in_light_region_flag = 1
                region_counter = 0
            else:
                region_counter = region_counter +1

    contour_pixels_line = np.zeros((rows,columns))

    #print("dark_region_contour_pixels_rowwise:",dark_region_contour_pixels_rowwise)

    contour_pixels = dark_region_contour_pixels_rowwise + dark_region_contour_pixels_colwise

    for i in contour_pixels:
        contour_pixels_line[i] = 1

    plt.matshow(contour_pixels_line, cmap="Blues")
    plt.savefig("Output/Contour/" + file_name + "_contour.png")











