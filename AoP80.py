# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 18:22:20 2023

@author: Tage
"""

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from tqdm import tqdm
from skimage.filters import gaussian, unsharp_mask
from skimage.filters import threshold_otsu, threshold_local
from astropy.visualization import astropy_mpl_style
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2
import scipy.signal as signal
from sklearn import preprocessing
import scipy.fft as scifft


promisepath = "C:/Users/joaki/Pictures/Q1-latest-whigal-85.fits"

#%%
plt.style.use(astropy_mpl_style)



def plot_general(function, dpi=100, fig_size=(30,13), title = None, vmin=None, vmax=None, cmap="gray_r", scale=1, grid=False, colorbar=False):
    mean = np.mean(function)
    std_dev = np.std(function)
    lower = mean - scale*std_dev
    upper = mean + scale*std_dev
    if vmin != None:
        vmin = function.min()
    elif vmin == 1:
        vmin = mean - 100*scale*std_dev
        
    if vmax != None:
        vmax = vmin = function.max()
    elif vmax == 1:
        mean + scale*std_dev
    
    plt.figure(figsize=fig_size, dpi=dpi)
    ax = plt.gca()
    im = ax.imshow(function, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    
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

    
    
    

def plot_figure(func, norm, title, dpi=False):
    plt.figure()
    if dpi:
        plt.rcParams['figure.dpi'] = 300
        
    
    plt.title(f"{title}")
    plt.grid(alpha=0.05)
    ax = plt.gca()
    #pcm = ax.pcolor(func, norm=colors.LogNorm(vmin=func.min(), vmax=func.max()))
    im = ax.imshow(func, norm=norm, cmap='gnuplot')
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
       
    plt.colorbar(im, cax=cax)
    plt.show()


def plot_figure_1(func, title, dpi=False):
    plt.figure()
    if dpi:
        plt.rcParams['figure.dpi'] = 300
        
    
    plt.title(f"{title}")
    plt.grid(alpha=0.05)
    ax = plt.gca()
    #pcm = ax.pcolor(func, norm=colors.LogNorm(vmin=func.min(), vmax=func.max()))
    im = ax.imshow(func, cmap='gnuplot')
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
       
    plt.colorbar(im, cax=cax)
    plt.show()

def plot_image(func, title, save=False, dpi=False):
    plt.figure()
    plt.axis('off')
    
    if dpi:
        plt.rcParams['figure.dpi'] = 300
    ax = plt.gca()
    im = ax.imshow(func, norm=colors.PowerNorm(gamma=0.5), cmap='gnuplot')
    
    
    if save:
        plt.savefig(f"{title}.png", dpi=300, bbox_inches=0)
    plt.show()
    

def unsharp_mask(function, sigma, k=1): 
    "k=1 unsharp mask, greater than 1, superfunction"
    gaus = gaussian(function, sigma=sigma)
    mask = function - gaus
    return ((function + k * mask), mask, gaus)


#%%
"Data split"

cutouts = []
xslice = 7500 #5000 #10000
yslice = 7000 #3500 #3500

with fits.open(promisepath, use_fsspec=True, fsspec_kwargs={"anon": True}) as hdul:  
    for i in tqdm(range(0, int(7000/yslice))):
        for j in tqdm(range(0, int(120000/xslice))):
            cutouts.append(hdul[0].section[yslice*i:yslice*i + yslice, xslice*j:xslice*j+xslice])
#%%
#(cutouts[1], dpi=300, colorbar=True, cmap="hot", title="Test plot 1", scale=20)

#%%
"Plotting cutouts"
#for i in range(0, len(cutouts)):  
    #plot_general(cutouts[i], dpi=300, colorbar=True, cmap="hot", title=f"Test plot {i}", scale=20)
    
#%%
"Creating unsharp masks"
#unsharp_masks = []
#masks = []
#gaussians = []
#for i in tqdm(range(0, len(cutouts))):
    #um, mask, gaus = unsharp_mask(cutouts[i], sigma=30, k=5)
    #unsharp_masks.append(um)
    #masks.append(mask)
    #gaussians.append(gaus)

#%%
"Plotting unsharp masks"
#for i in range(0, len(cutouts)):  
   # plot_general(unsharp_masks[i], dpi=300, colorbar=True, vmin=None, vmax=1, cmap="hot", title=f"Test plot {i}", scale=20)
    
#%%
"Creating different Gausians"


#%%
"Work in prgress, är nog lite knas"
#threshhold_global = []
#threshhold_global_mask = []
#for i in tqdm(range(0, len(unsharp_masks))):
    #data = cutouts[i]
    #mask = unsharp_masks[i]
    
    #threshhold = threshold_otsu(mask, nbins=256)
    #result_mask = ((mask) > threshhold)#* mask
    #result_mask_1 = ((mask) > threshhold)* mask
    
    #plot_figure_1(data, "global threshhold")
    #plot_figure_1(result_mask/np.linalg.norm(result_mask), "global threshold on mask", True)
    #threshhold_global_mask.append(result_mask)

#%%
"Work in prgress, är nog lite knas"

#threshhold_local = []
#threshhold_local_mask = []
#for i in tqdm(range(0, 15)):
    #data = cutouts[i]
    #mask = unsharp_masks[i]
    #b = threshold_local(data, block_size=3, method='gaussian', offset=0, mode='reflect', param=None, cval=0)
    #result = ((data) > b) * data
    
    #c = threshold_local(mask, block_size=3, method='gaussian', offset=-1, mode='reflect', param=None, cval=0)
    #result_mask = ((mask) > c)
    
    #plot_figure(data, colors.Normalize(0,1), "Initial data")
    #plot_figure(mask, colors.Normalize(0,1), "Unsharp mask")
    #plot_figure(result, "local threshhold on data")
    #plot_figure(result_mask, colors.Normalize(0,1), "local threshold on mask", True)
    #plot_image(result_mask/np.linalg.norm(result_mask), "test threshholding", False, True)
    #threshhold_local.append(result)
    #threshhold_local_mask.append(np.linalg.norm(result_mask))
    
    
    #%% Fourier transformer

def plot_spectrum(im_fft):
    from matplotlib.colors import LogNorm
    # A logarithmic colormap
    plt.imshow(np.abs(im_fft), norm=LogNorm(vmin=5))
    plt.colorbar()

plot_general(cutouts[6], dpi=300, colorbar=True, cmap="hot", title="Cutouts 6 original", scale=20)
    
fourier2d = scifft.rfft2(cutouts[1], s=None, axes=(-2, -1), norm=None)

#plot_general(fourier2d, dpi=300, colorbar=True, cmap="hot", title="Test plot Fourier 1", scale=20)

plot_spectrum(fourier2d)
plt.title("Fourier Transform")

fraction_keep = 0.01
fouriercopy = fourier2d
row, column = fourier2d.shape
fouriercopy[int(row*fraction_keep):int(row*(1-fraction_keep))] = 0
fouriercopy[:, int(column*fraction_keep):int(column*(1-fraction_keep))] = 0

plt.figure()
plot_spectrum(fouriercopy)
plt.title("Filtered Fourier Transform")

fourier2d = scifft.irfft2(fouriercopy, s=None, axes=(-2, -1), norm=None).real
#plot_general(fourier2d, dpi=300, colorbar=True, cmap="hot", title="Test plot fourier 2", scale=20)

plot_general(fourier2d, dpi=300, colorbar=True, cmap="hot", title="recunstructed image", scale=20)



        
