# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 11:05:19 2023

@author: joaki
"""
import skimage
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
import scipy.fft as scifft
import scipy.signal as signal
from sklearn import preprocessing


promisepath = "C:/Users/joaki/Pictures/Q1-latest-whigal-85.fits"
hdul = fits.open(promisepath)

fitspromise = hdul[0].data[0:10000, 0:10000]




fourier2d = scifft.fft2(fitspromise, s=None, axes=(-2, -1), norm=None)
plt.plot(fitspromise)
plt.imshow(fitspromise)

#Skiten är inte i närheten av rimligt/fungerande