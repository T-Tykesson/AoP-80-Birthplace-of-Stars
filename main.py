import matplotlib.pyplot as plt

from plotting import *

from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.visualization import astropy_mpl_style

from skimage.filters import gaussian, unsharp_mask, threshold_otsu, threshold_local

import numpy as np
from tqdm import tqdm
import scipy.fft as scifft
from scipy.ndimage import gaussian_filter

from enum import Enum
from typing import Tuple


# Notes:
"""
The data is not square, the top and bottom edges are spikey-ish, making auto-detection of scale tricky
A log colour scale works better than regular
Data can be slices easily as it is a numpy array, using memmap is probably best
The lab computers can load the entire image in about 3 mins
There is only one HDU (the primary one), we could add our masks to the same FITS file
"""


# Functions using this decorator can either be called directly, or called via the .plot(<input>) attribute and be plotted simultaniously
def plottable(fig_index, **fig_kwargs):
    def plot_func(func_to_plot, *args, **kwargs):
        ret_val = func_to_plot(*args, **kwargs)
        plot_general(ret_val, fig_index=fig_index, **fig_kwargs)
        return ret_val

    def decorator(func):
        func.plot = lambda *args, **kwargs: plot_func(func, *args, **kwargs)
        return func
    return decorator


class SeedClassifier:
    def __init__(self, promise_path: str, limits: Tuple, single_slice=True):

        self.cutouts = []

        with fits.open(promise_path, memmap=True) as fits_data:
            
            xmax, ymax = fits_data["PRIMARY"].data.shape

            if not single_slice:

                xslice_size = limits[0]
                yslice_size = limits[1]

                for i in tqdm(range(0, int(ymax/yslice_size))):
                    for j in tqdm(range(0, int(xmax/xslice_size))):
                        self.cutouts.append(fits_data["PRIMARY"].data[yslice_size*i : yslice_size*i+yslice_size, xslice_size*j : xslice_size*j+xslice_size])
            else:
                self.cutouts.append(fits_data["PRIMARY"].data[ limits[0]:limits[1], limits[2]:limits[3] ])
    
    def run(self):
        
        for i, cutout in enumerate(tqdm(self.cutouts)):
            plot_general(cutout, title=f"Original {i+1}", scale=40, fig_index=1)

            self.unsharp_mask_basic.plot(cutout)
            
        plt.show()

    @plottable(fig_index=9, title="Basic unsharp masking")
    def unsharp_mask_basic(img, kernel_size=1, weight=1.0, hi_threshold=None):
        gaus = gaussian(img, sigma=kernel_size)
        mask = img - gaus
        applied = img + weight * mask
        if hi_threshold:
            applied[applied >= hi_threshold] = hi_threshold
        return mask, applied


    def unsharp_mask_fft(self, img, kernel_size=1, weight=1, lo_threshold=0.02, hi_threshold=None):
        img_freqs = scifft.fft2(img)
        return None
        """
        gaus_low_freq_filter_mask = gaussian_filter(input, sigma, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0, *, radius=None)

        D0 = 10
        for u in range(M):
            for v in range(N):
                D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
                H[u,v] = np.exp(-D**2/(2*D0*D0))
        """

    def low_pass_filter_fourier(self, img, lo_threshold_percentage=0.02):
        
        fourier2d = scifft.fft2(img)

        # plot_general(fourier2d, dpi=300, colorbar=True, cmap="hot", title="Test plot Fourier 1", scale=20)

        plt.title("Fourier Transform")
        plot_spectrum(fourier2d)

        fouriercopy = fourier2d.copy()
        row, column = fourier2d.shape
        fouriercopy[int(row*lo_threshold_percentage):int(row*(1-lo_threshold_percentage))] = 0
        fouriercopy[:, int(column*lo_threshold_percentage):int(column*(1-lo_threshold_percentage))] = 0

        plt.figure()
        plot_spectrum(fouriercopy)
        plt.title("Low Filtered Fourier Transform")

        fourierfiltered = scifft.ifft2(fouriercopy).real

        plot_general(fourierfiltered, dpi=300, colorbar=True, cmap="hot", title="Reconstructed image", scale=20)


if __name__ == "__main__":
    plt.style.use(astropy_mpl_style)
    
    src_path = "src_data/PROMISE-Q1-8micron-filled-v0_3.fits"

    X_LOWER, X_UPPER = 118_300, 118_900
    Y_LOWER, Y_UPPER = 8_400, 9_000

    sc = SeedClassifier(src_path, [Y_LOWER, Y_UPPER, X_LOWER, X_UPPER])
    sc.run()

"""
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
"""