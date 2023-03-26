# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 17:16:24 2023

@author: Tage
"""

import detection
import definition
import get_data
import artificial_cores
import plotting
import us_mask
from tqdm import tqdm 
import numpy as np

#%%
file_path = "C:/Users/Tage/Programmering/AoP80/Q1-latest-whigal-85.fits"
xslice = 5000
yslice = 7000

data_list = get_data.get_splitted_data(file_path, yslice, xslice)
#%%
"lägger till cores i data"
half_data = data_list[0:3]
art_data, art_cores, art_tuples = artificial_cores.insert_art_cores(half_data, xslice*8, amount=2000)
#%%
"här lägger man in metod + definition som ger mask"
binary_mask_list = []
gaussian_blur_list = []
final_mask_list = []


for data in tqdm(art_data):
    g_blur, binary = us_mask.gaussian_threshold(data, 25, 10)
    final_mask = np.full(np.shape(g_blur), False)

    plotting.plot(g_blur)
    plotting.plot(binary)
    
    sigma = 10
    check_distance = 50
    mult = 2
    lower_peak_val = 5
    
    rows, cols = np.where(binary == True)
    g_test_left = np.transpose([g_blur[rows,np.minimum((cols + i), [len(data[0]) - 1]*len(cols))] for i in range(-check_distance, 0)])
    g_test_right = np.transpose([g_blur[rows,np.minimum((cols + i), [len(data[0]) - 1]*len(cols))] for i in range(0, check_distance)])
    print("Retrieved surrounding pixels")
    padded = np.expand_dims(data[rows, cols],1)
    padded = np.pad(padded, [[0, 0], [0, check_distance - 1]], 'edge')
    print("Padding done")
    g_check_left = padded > g_test_left * mult
    g_check_left = np.any(g_check_left, axis=1)
    print("Check left done")
    g_check_right = padded > g_test_right * mult
    g_check_right = np.any(g_check_right, axis=1)
    print("Check right done")
    #g_check = np.bitwise_and([True]*len(g_check_right), g_check_right)
    g_check = np.bitwise_and(g_check_left, g_check_right)
    g_peak_check = data[rows, cols] > lower_peak_val
    print("Check peak done")
    g_check = np.bitwise_and(g_check, g_peak_check)
    indicies = np.transpose([rows[g_check], cols[g_check]])
    print("Indicies done")
    final_mask[rows[g_check], cols[g_check]] =True
    
    plotting.plot(final_mask)
    final_mask_list.append(final_mask)
    binary_mask_list.append(binary)
    gaussian_blur_list.append(g_blur)
    
#%%
"Kör sin mask genom detection för att få ut koordinater"
detection_list, detection_list_combined, found_tuple_list = detection.log_detection_multiple(final_mask_list, xslice*8)
#tar ca 5-10 min att beräkna en slice på x = 10000, y = 7000
 
#%%
found, percentage = artificial_cores.test_cores(art_tuples, found_tuple_list)
print("Percentage: ", percentage)

#%%
plotting.blob_plot_multiple(detection_list, art_data, ylow=0, yhigh=7000, xlow=0, xhigh=5000, dpi=100)

#%%
plotting.blob_plot_multiple(detection_list, final_mask_list, ylow=0, yhigh=7000, xlow=0, xhigh=5000, dpi=100)
