# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 09:40:49 2023

@author: Tage
"""

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from tqdm import tqdm
from skimage.filters import gaussian, unsharp_mask
from skimage.filters import threshold_otsu, threshold_local
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.exposure import rescale_intensity
from astropy.visualization import astropy_mpl_style
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2
import scipy.signal as signal
from sklearn import preprocessing

from skimage.filters import gaussian, threshold_local
from skimage.feature import blob_dog, blob_log, blob_doh
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from astropy.io import fits
import numpy as np
from sys import exit
import time

#%%
full_data = True #can edit to False for part of data
file_path = "Q1-latest-whigal-85.fits"

if full_data:
    full_data_list = []
    xslice = 10000
    yslice = 7000

    with fits.open("Q1-latest-whigal-85.fits", use_fsspec=True, fsspec_kwargs={"anon": True}) as hdul:  
       #cutout = hdul[0].section[0:1750, 0:2500] 
        for i in tqdm(range(0, int(7000/yslice))):
            for j in tqdm(range(0, int(120000/xslice))):
                cutout = hdul[0].section[yslice*i:yslice*i + yslice, xslice*j:xslice*j+xslice]
                full_data.append(cutout)
else:
    data = fits.getdata(file_path)
    x_low = 0
    x_high = 20000
    y_low = 0
    y_high = 7000
    data = data[y_low:y_high, x_low:x_high]

#%%
def gaussian_threshold(data, sigma, offset):
    g = gaussian(data, sigma=sigma)
    binary = data > (g + offset)
    return g, binary

def imshow(data, norm_low, norm_high, cmap="hot", dpi=15, figsize=(50, 50)):
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(data, cmap=cmap, origin='lower', interpolation="none", norm=Normalize(norm_low, norm_high))
    plt.show()

def plot(function, dpi=100, fig_size=(30,13), title = None, norm=None, vmin=None, vmax=None, cmap="gray_r", scale=1, grid=False, colorbar=False):
    mean = np.mean(function)
    std_dev = np.std(function)
    lower = mean - scale*std_dev
    upper = mean + scale*std_dev

    plt.figure(figsize=fig_size, dpi=dpi)
    ax = plt.gca()
    im = ax.imshow(function, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
    
    if title != None: 
        plt.title(f"{title}")
    
    if grid:
        plt.grid(alpha=0.05)
    else:
        plt.grid(alpha=0)
        
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
    plt.show()


#%%

"Original, normal mask"
if full_data:
    g_blur_list = []
    final_mask_list = []
    
    for data in tqdm(full_data):
        g_blur, _ = gaussian_threshold(data, 10, 1)
        final_mask = np.full(np.shape(g_blur), False)
        
        sigma = 10
        check_distance = 50
        mult = 2
        lower_peak_val = 5
        
        g, binary = gaussian_threshold(data, sigma, 1)
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
        
        g_blur_list.append(g_blur)
        final_mask_list.append(final_mask)
        
        imshow(g_blur, 0, 70)
        imshow(binary, 0, 1)
        imshow(data, 0, 70)
        imshow(final_mask, 0, 1)  
        
else:
    g_blur, _ = gaussian_threshold(data, 10, 1)
    final_mask = np.full(np.shape(g_blur), False)
    
    sigma = 10
    check_distance = 50
    mult = 2
    lower_peak_val = 5
    
    g, binary = gaussian_threshold(data, sigma, 1)
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
    
    imshow(g_blur, 0, 70)
    imshow(binary, 0, 1)
    imshow(data, 0, 70)
    imshow(final_mask, 0, 1)
    
#%%

def blob_log_detection(function, max_sigma=30, num_sigma=10, threshold=.1): #Laplacian of Gaussian
    blobs_log = blob_log(function, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)
    blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2) # Compute radii in the 3rd column. 
    return blobs_log

#def blob_dog_detection(function, max_sigma=30,  threshold=.1): #Difference of Gaussian
#    blobs_dog = blob_dog(function, max_sigma=max_sigma, threshold=threshold)
#    blobs_dog[:, 2] = blobs_dog[:, 2] * np.sqrt(2) # Compute radii in the 3rd column. 
#    return blobs_dog

#def blob_doh_detection(function, min_sigma=0, max_sigma=300, num_sigma=30, threshold=50): #Difference of Hessian
#    blobs_log = blob_doh(function, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)
#    return blobs_log

#%%
# funkar nu bara om yslice är 7000 (hela längden på )
if full_data:
    detection_list = []
    k = 0
    for final_mask in tqdm(final_mask_list):
        # x är på dim 1, av 0,1,2
        blob_log_list = blob_log_detection(final_mask)
        detection_list.append(blob_log_list)
        if k != 0:
            for i in leng(blob_log_list):
            blob_log_list[:,1] = blob_log_list[:,1] + k * xslice
        k + = 1
else:
    blob_log_list = blob_log_detection(final_mask)

#%%

"plotting"

if full_data:
    k = 0
    for blob_log_list in detection_list:
        title=f"laplacian of gaussian {i}"
        
        
        fig, ax = plt.subplots(figsize=(90, 30), sharex=True, sharey=True)
        ax.set_title(title)
        ax.imshow(full_data[k], cmap="inferno", origin="lower", vmin=(np.mean(data)-1*np.std(data)), vmax=(np.mean(data)+4*np.std(data)))
        #ax.imshow(final_mask, origin="lower", norm=Normalize(0, 1))
        
        for blob in blob_log_list:
            y, x, r = blob
            c = plt.Circle((x, y), 4*r, color="red", linewidth=2, fill=False)
            ax.add_patch(c)
        ax.set_axis_off()
         
        plt.tight_layout()
        plt.show()
        k += 1
        
else:
    fig, ax = plt.subplots(figsize=(90, 30), sharex=True, sharey=True)
    ax.set_title(title)
    ax.imshow(data, cmap="inferno", origin="lower", vmin=(np.mean(data)-1*np.std(data)), vmax=(np.mean(data)+4*np.std(data)))
    #ax.imshow(final_mask, origin="lower", norm=Normalize(0, 1))
        
    for blob in blob_log_list:
        y, x, r = blob
        c = plt.Circle((x, y), 4*r, color="red", linewidth=2, fill=False)
        ax.add_patch(c)
        ax.set_axis_off()
          
    plt.tight_layout()
    plt.show()
    





