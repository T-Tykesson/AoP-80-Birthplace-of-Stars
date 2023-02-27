import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.visualization import astropy_mpl_style

from skimage.filters import gaussian, unsharp_mask, threshold_otsu, threshold_local

import numpy as np
from tqdm import tqdm

plt.style.use(astropy_mpl_style)
X_LOWER, X_UPPER = 0, 120_000
Y_LOWER, Y_UPPER = 0, 12_000

# Notes:
"""
The data is not square, the top and bottom edges are spikey-ish, making auto-detection of scale tricky
A log colour scale works better than regular
Data can be slices easily as it is a numpy array, using memmap is probably best
The lab computers can load the entire image in about 3 mins
There is only one HDU (the primary one), we could add our masks to the same FITS file
"""

#%%
"Data split"

src_path = "src_data/PROMISE-Q1-8micron-filled-v0_3.fits"

use_cutouts = False  # Easier on the memory if we just use one slice
cutouts = []
xslice = 7500 #5000 #10000
yslice = 7000 #3500 #3500

X_LOWER, X_UPPER = 115_000, 120_000
Y_LOWER, Y_UPPER = 7_000, 9_000


with fits.open(src_path, memmap=True) as fits_data:  # use_fsspec=True, fsspec_kwargs={"anon": True}, memmap=True) as hdul: 
    if use_cutouts:
        for i in tqdm(range(0, int(7000/yslice))):
            for j in tqdm(range(0, int(120000/xslice))):
                cutouts.append(fits_data[0].section[yslice*i:yslice*i + yslice, xslice*j:xslice*j+xslice])
    else:
        cutouts.append(fits_data["PRIMARY"].data[ Y_LOWER:Y_UPPER, X_LOWER:X_UPPER])

#%%

def plot_general(function, dpi=60, fig_size=(30,13), title = None, vmin=None, vmax=None, cmap="gray_r", scale=1, grid=False, colorbar=False):
    mean = np.mean(function)
    std_dev = np.std(function)
    lower = mean - scale*std_dev
    upper = mean + scale*std_dev
    if vmin == None:
        vmin = function.min()
    elif vmin == 1:
        vmin = mean - scale*std_dev
        
    if vmax == None:
        vmax = function.max()
    elif vmax == 1:
        vmax = mean + scale*std_dev
    
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


def plot_figure(func, title, norm=None, dpi=None):
    plt.figure()
    if dpi:
        plt.rcParams['figure.dpi'] = dpi
        
    
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


def plot_image(func, title, save=False, dpi=None):
    plt.figure()
    plt.axis('off')
    
    if dpi:
        plt.rcParams['figure.dpi'] = dpi
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
"Plot cutouts"
for i in range(0, len(cutouts)):  
    plot_general(cutouts[i], colorbar=True, cmap="hot", title=f"Plot raw {i}", scale=20)
    
#%%
"Create unsharp masks"
unsharp_masks = []
masks = []
gaussians = []
for i in tqdm(range(0, len(cutouts))):
    um, mask, gaus = unsharp_mask(cutouts[i], sigma=30, k=5)
    unsharp_masks.append(um)
    masks.append(mask)
    gaussians.append(gaus)

#%%
"Plot unsharp masks"
for i in range(0, len(cutouts)):  
    plot_general(unsharp_masks[i], colorbar=True, vmin=None, vmax=1, cmap="hot", title=f"Unsharp mask {i}", scale=20)
    
#%%
"Creating different Gausians"


#%%
"Work in prgress, är nog lite knas"
threshhold_global = []
threshhold_global_mask = []
for i in tqdm(range(0, len(unsharp_masks))):
    data = cutouts[i]
    mask = unsharp_masks[i]
    
    threshhold = threshold_otsu(mask, nbins=256)
    result_mask = ((mask) > threshhold)#* mask
    result_mask_1 = ((mask) > threshhold)* mask
    
    plot_figure(data, "global threshhold")
    plot_figure(result_mask/np.linalg.norm(result_mask), "global threshold on mask")
    threshhold_global_mask.append(result_mask)

#%%
"Work in prgress, är nog lite knas"

threshhold_local = []
threshhold_local_mask = []
for i in tqdm(range(0, len(unsharp_masks))):
    data = cutouts[i]
    mask = unsharp_masks[i]
    #b = threshold_local(data, block_size=3, method='gaussian', offset=0, mode='reflect', param=None, cval=0)
    #result = ((data) > b) * data
    
    c = threshold_local(mask, block_size=3, method='gaussian', offset=-1, mode='reflect', param=None, cval=0)
    result_mask = ((mask) > c)
    
    plot_figure(data, "Initial data", colors.Normalize(0,1))
    plot_figure(mask, "Unsharp mask", colors.Normalize(0,1))
    #plot_figure(result, "local threshhold on data")
    plot_figure(result_mask, "local threshold on mask", colors.Normalize(0,1),)
    plot_image(result_mask/np.linalg.norm(result_mask), "test threshholding")
    #threshhold_local.append(result)
    #threshhold_local_mask.append(np.linalg.norm(result_mask))
