# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 16:49:42 2023

@author: Tage
"""

import numpy as np
from tqdm import tqdm
from skimage.feature import blob_dog, blob_log, blob_doh

def blob_log_detection(function, max_sigma=30, num_sigma=10, threshold=.1): #Laplacian of Gaussian
    blobs_log = blob_log(function, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)
    blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2) # Compute radii in the 3rd column. 
    return blobs_log

def blob_dog_detection(function, max_sigma=30,  threshold=.1): #Difference of Gaussian
    blobs_dog = blob_dog(function, max_sigma=max_sigma, threshold=threshold)
    blobs_dog[:, 2] = blobs_dog[:, 2] * np.sqrt(2) # Compute radii in the 3rd column. 
    return blobs_dog

def blob_doh_detection(function, min_sigma=0, max_sigma=300, num_sigma=30, threshold=50): #Difference of Hessian
    blobs_log = blob_doh(function, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)
    return blobs_log

def log_detection_multiple(data_list, xslice, max_sigma=30, num_sigma=10, threshold=.1): #Laplacian of Gaussian
    detection_list = []
    k = 0
    for data in tqdm(data_list): # x är på dim 1, av 0,1,2
        blob_log_list = blob_log_detection(data)
        if k != 0:
            blob_log_list[:,1] = blob_log_list[:,1] + k * xslice
        detection_list.append(blob_log_list)
        k += 1
    detection_list = np.array(detection_list)
    detection_list_combined = np.concatenate(detection_list, axis=0)
    found_tuple_list =  tuple(map(tuple, detection_list_combined[:,0:2]))
    return detection_list, detection_list_combined, found_tuple_list