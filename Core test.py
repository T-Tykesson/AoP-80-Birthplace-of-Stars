# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 09:41:15 2023

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
full_data = True #Ändra till false om man inte vill dela upp hela datan
file_path = "Q1-latest-whigal-85.fits"

if full_data:
    data_list = []
    xslice = 10000
    yslice = 7000

    with fits.open("Q1-latest-whigal-85.fits", use_fsspec=True, fsspec_kwargs={"anon": True}) as hdul:  
       #cutout = hdul[0].section[0:1750, 0:2500] 
        for i in tqdm(range(0, int(7000/yslice))):
            for j in tqdm(range(0, int(120000/xslice))):
                cutout = hdul[0].section[yslice*i:yslice*i + yslice, xslice*j:xslice*j+xslice]
                data_list.append(cutout)
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

def create_gaussian_filter(kernel_size, sigma=1, muu=0):
    # Initializing value of x,y as grid of kernel size
    # in the range of kernel size
    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),
                       np.linspace(-1, 1, kernel_size))
    dst = np.sqrt(x**2+y**2)
    # lower normal part of gaussian
    normal = 1/(2.0 * np.pi * sigma**2)
    # Calculating Gaussian filter
    gauss = np.exp(-((dst-muu)**2 / (2.0 * sigma**2))) * normal
    return gauss

"Artificial star test, all same size"

"Create circular mask"
def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

"Insert circles into data"
def insert_circles(data, gaussian, nr): #antar symetrisk gaussian
    size = len(gaussian)
    art_catalog = []
    data_copy = np.array(data, copy=True)
    for i in range(nr):
        xrandom = np.random.randint(0, high=len(data[1])-len(gaussian)) #ändra 
        yrandom = np.random.randint(0, high=len(data)-len(gaussian))
        
        intensity = np.random.randint(50, 350)
        
        art_catalog.append((int(yrandom+size/2), int(xrandom+size/2), intensity)) #want to get middle of point
        
        data_copy[yrandom:yrandom+size,xrandom:xrandom+size] += intensity*gaussian
    
    #plot(data, dpi=300, colorbar=True, title="test", vmin=np.mean(data)-1*np.std(data), vmax=(np.mean(data)+6*np.std(data)), cmap="inferno")     
    #plot(data_copy, dpi=300, colorbar=True, title="test", vmin=np.mean(data)-1*np.std(data), vmax=(np.mean(data)+6*np.std(data)), cmap="inferno")   
    return data_copy, art_catalog

#%%
if full_data:
    art_data_list = []
    art_catalog_list = []
    
    kernel_size=20
    amount = 100
    gaussian_core = create_gaussian_filter(kernel_size)
    h, w = gaussian_core.shape[:2]
    circular_mask = create_circular_mask(h, w)
    
    art_core = gaussian_core.copy()
    art_core[~circular_mask] = 0
    
    k = 0  
    for data in tqdm(data_list):
        art_data, art_catalog = insert_circles(data, art_core, amount)
        art_catalog = np.array(art_catalog)
        if k != 0:
            art_catalog[:,1] = art_catalog[:,1] + k * xslice
        art_data_list.append(art_data)
        art_catalog_list.append(art_catalog)
        k += 1
    
    art_catalog_list = np.array(art_catalog_list).reshape(amount*int(120000/xslice), 3)

else:
    kernel_size=20
    gaussian_core = create_gaussian_filter(kernel_size)
    h, w = gaussian_core.shape[:2]
    circular_mask = create_circular_mask(h, w)
    art_core = gaussian_core.copy()
    art_core[~circular_mask] = 0
    
    art_data, art_catalog = insert_circles(data, art_core, 1000) #sista värdet är hur många man vill ha i datan

#%%


#%%

"Enkel mask, gammal metod kollar vänster/höger gjort så kan köras på hela datan eller delar"
if full_data:
    g_blur_list = []
    final_mask_list = []
    
    for data in tqdm(art_data_list):
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
        
        #imshow(g_blur, 0, 70)
        #imshow(binary, 0, 1)
        #imshow(data, 0, 70)
        #imshow(final_mask, 0, 1)  
        
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
    
    #imshow(g_blur, 0, 70)
    #imshow(binary, 0, 1)
    #imshow(data, 0, 70)
    #imshow(final_mask, 0, 1)
    
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
if full_data:
    detection_list = []
    for final_mask in tqdm(final_mask_list):
        blob_log_list = blob_log_detection(final_mask)
        detection_list.append(blob_log_list)
else:
    blob_log_list = blob_log_detection(final_mask)

#%%
detection_list = np.array(detection_list)
detection_list = np.concatenate(detection_list, axis=0)

#%%

"plotting"

if full_data:
    k = 0
    for blob_log_list in detection_list:
        title=f"laplacian of gaussian {i}"
        k += 1
        
        fig, ax = plt.subplots(figsize=(90, 30), sharex=True, sharey=True)
        ax.set_title(title)
        ax.imshow(art_data_list[0], cmap="inferno", origin="lower", vmin=(np.mean(data)-1*np.std(data)), vmax=(np.mean(data)+4*np.std(data)))
        #ax.imshow(final_mask, origin="lower", norm=Normalize(0, 1))
        
        for blob in blob_log_list:
            y, x, r = blob
            c = plt.Circle((x, y), 4*r, color="red", linewidth=2, fill=False)
            ax.add_patch(c)
        ax.set_axis_off()
        
        #plt.xlim(3000, 4000) #choose x-region
        #plt.ylim(1500, 2000) #choose y-region
        plt.tight_layout()
        plt.show()
    
    for blob_log_list in detection_list:
        title=f"laplacian of gaussian {i}"
        k += 1
        
        fig, ax = plt.subplots(figsize=(90, 30), sharex=True, sharey=True)
        ax.set_title(title)
        #ax.imshow(data, cmap="inferno", origin="lower", vmin=(np.mean(data)-1*np.std(data)), vmax=(np.mean(data)+4*np.std(data)))
        ax.imshow(final_mask_list[0], origin="lower", norm=Normalize(0, 1))
        
        for blob in blob_log_list:
            y, x, r = blob
            c = plt.Circle((x, y), 4*r, color="red", linewidth=2, fill=False)
            ax.add_patch(c)
        ax.set_axis_off()
        
        #plt.xlim(3000, 4000) #choose x-region
        #plt.ylim(1500, 2000) #choose y-region
        plt.tight_layout()
        plt.show()
        
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

#%%
if full_data:
    art_tuple_list = tuple(map(tuple, art_catalog_list[:,0:2]))
    found_tuple_list =  tuple(map(tuple, detection_list[:,0:2]))
else: #inte testad kan behöva ändras
    art_tuple_catalog = tuple(map(tuple, art_catalog[:,0:2])
    detection_tuple  = tuple(map(tuple, detection_list[:,0:2]))

#%%

found_list = []
if full_data:
    for tup in art_tuple_list: 
        if tup in found_tuple_list:
            found_list.append(tup)
else: #inte testad kan behöva ändras
    for tup in art_tuple_catalog: 
        if tup in detection_tuple:
            found_list.append(tup)







