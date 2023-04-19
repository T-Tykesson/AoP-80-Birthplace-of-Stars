# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 17:20:18 2023

@author: Tage

core test
"""
import numpy as np
from tqdm import tqdm

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
def insert_circles(data, gaussian, nr, intensity="Random", int_min=50, int_max=350): #antar symetrisk gaussian
    size = len(gaussian)
    art_catalog = []
    data_copy = np.array(data, copy=True)
    for i in range(nr):
        
        xrandom = np.random.randint(0, high=len(data[1])-len(gaussian)) #ändra 
        yrandom = np.random.randint(0, high=len(data)-len(gaussian))
        
        if intensity == "Random":
            inten = np.random.randint(int_min, int_max)
        else: 
            inten = intensity

        art_catalog.append([int(yrandom+size/2), int(xrandom+size/2), inten]) #want to get middle of point
        
        data_copy[yrandom:yrandom+size,xrandom:xrandom+size] += inten*gaussian
    
    #plot(data, dpi=300, colorbar=True, title="test", vmin=np.mean(data)-1*np.std(data), vmax=(np.mean(data)+6*np.std(data)), cmap="inferno")     
    #plot(data_copy, dpi=300, colorbar=True, title="test", vmin=np.mean(data)-1*np.std(data), vmax=(np.mean(data)+6*np.std(data)), cmap="inferno")   
    return data_copy, art_catalog

def insert_circles_different_sizes(data, gaussian_list, nr, intensity="Random", int_min=50, int_max=350): #antar symetrisk gaussian
    art_catalog = []
    data_copy = np.array(data, copy=True)
    for i in range(nr-1):
        size = len(gaussian_list[i])
        
        xrandom = np.random.randint(0, high=len(data[1])-len(gaussian_list[i])) 
        yrandom = np.random.randint(0, high=len(data)-len(gaussian_list[i]))
        
        if intensity == "Random":
            inten = np.random.randint(int_min, int_max)
        else: 
            inten = intensity

        art_catalog.append([int(yrandom+size/2), int(xrandom+size/2), inten, size/2]) #want to get middle of point
        
        data_copy[yrandom:yrandom+size,xrandom:xrandom+size] += inten*gaussian_list[i]
    
    #plot(data, dpi=300, colorbar=True, title="test", vmin=np.mean(data)-1*np.std(data), vmax=(np.mean(data)+6*np.std(data)), cmap="inferno")     
    #plot(data_copy, dpi=300, colorbar=True, title="test", vmin=np.mean(data)-1*np.std(data), vmax=(np.mean(data)+6*np.std(data)), cmap="inferno")   
    return data_copy, art_catalog

def insert_art_cores_data_slices(data_list, xslice, kernel_size=20, amount=1000, intensity="Random", int_min=50, int_max=350):
    art_data_list = []
    art_catalog_list = []
    
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
    art_catalog_tuples = tuple(map(tuple, art_catalog_list[:,0:2]))
    
    return art_data_list, art_catalog_list, art_catalog_tuples

def insert_art_cores(data, kernel_size=20, amount=1000, intensity="Random", int_min=50, int_max=350):
    gaussian_core = create_gaussian_filter(kernel_size)
    h, w = gaussian_core.shape[:2]
    circular_mask = create_circular_mask(h, w)
    
    art_core = gaussian_core.copy()
    art_core[~circular_mask] = 0
    
    art_data, art_catalog = insert_circles(data, art_core, amount, intensity=intensity, int_min=int_min, int_max=int_max)
    art_catalog = np.array(art_catalog)
    #art_catalog_list.append(art_catalog)

    art_catalog_tuples = list(map(tuple, art_catalog[:,0:2]))
    
    return art_data, art_catalog, art_catalog_tuples

def insert_art_cores_2(data, size_min=5, size_max=130, amount=1000, intensity="Random", int_min=50, int_max=350):
    art_core_list=[]
    for i in range(amount):
        kernel_size = np.random.randint(size_min, size_max)
        gaussian_core = create_gaussian_filter(kernel_size)
        h, w = gaussian_core.shape[:2]
        circular_mask = create_circular_mask(h, w)
        
        art_core = gaussian_core.copy()
        art_core[~circular_mask] = 0
        art_core_list.append(art_core)
    
    art_data, art_catalog = insert_circles_different_sizes(data, art_core_list, amount, intensity=intensity, int_min=int_min, int_max=int_max)
    art_catalog = np.array(art_catalog)
    #art_catalog_list.append(art_catalog)

    art_catalog_tuples = list(map(tuple, art_catalog[:,0:2]))
    
    return art_data, art_catalog, art_catalog_tuples

def test_cores(art_catalog_tuples, found_catalog_tuples): # input är tuples av koordinater
    found = []
    for tup in art_catalog_tuples:
        if tup in found_catalog_tuples:
            found.append(tup)
            
    percentage = len(found)/len(art_catalog_tuples)
    return found, percentage

def create_art_artefacts(kernel_size = 51, a = 1/9): #artefacter är 50 stora ish
    x = np.linspace(-kernel_size//2, kernel_size//2, kernel_size-1)
    y = np.linspace(-kernel_size//2, kernel_size//2, kernel_size-1)
    X, Y = np.meshgrid(x, y)

    sinc2d = np.sinc(a*np.hypot(X, Y))
    return sinc2d

def insert_art_artefacts(data, amount, intensity="Random", int_min=50, int_max=350):
    sinc2d = create_art_artefacts()
    h, w = sinc2d.shape[:2]
    circular_mask = create_circular_mask(h, w)
    
    art_artefact = sinc2d.copy()
    art_artefact[~circular_mask] = 0
    art_data, art_catalog = insert_circles(data, art_artefact, amount, intensity=intensity, int_min=int_min, int_max=int_max)
    art_catalog = np.array(art_catalog)

    art_catalog_tuples = list(map(tuple, art_catalog[:,0:2]))
    
    return art_data, art_catalog, art_catalog_tuples