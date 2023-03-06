import matplotlib.pyplot as plt
import matplotlib.colors as colors

from plotting import *

from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.visualization import astropy_mpl_style

from skimage.filters import gaussian, unsharp_mask, threshold_otsu, threshold_local

import numpy as np
from tqdm import tqdm

from typing import Tuple


# Notes:
"""
The data is not square, the top and bottom edges are spikey-ish, making auto-detection of scale tricky
A log colour scale works better than regular
Data can be slices easily as it is a numpy array, using memmap is probably best
The lab computers can load the entire image in about 3 mins
There is only one HDU (the primary one), we could add our masks to the same FITS file
"""



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
            print(cutout)
            plot_general(cutout, colorbar=True, title=f"Original {i+1}", scale=40)

            plot_general(self.unsharp_mask_basic(cutout), title=f"Applied unsharp masking {i+1}")
        
            # plot_general(unsharp_masks[i], colorbar=True, vmin=None, vmax=1, cmap="hot", title=f"Unsharp mask {i}", scale=20)

            # plot_general(self.unsharp_mask_fft())
            # plot_general(self.highpass())
            # plot_general(self.lowpass())

    def unsharp_mask_basic(self, img, kernel_size=1, weight=1, threshold=None):
        gaus = gaussian(img, sigma=kernel_size)
        mask = img - gaus
        applied = img + weight * mask
        if threshold:
            applied[applied >= threshold] = threshold
        return applied


if __name__ == "__main__":
    plt.style.use(astropy_mpl_style)
    
    src_path = "src_data/PROMISE-Q1-8micron-filled-v0_3.fits"

    X_LOWER, X_UPPER = 115_000, 120_000
    Y_LOWER, Y_UPPER = 7_000, 9_000

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