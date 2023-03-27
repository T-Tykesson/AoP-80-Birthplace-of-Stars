from skimage.filters import gaussian, threshold_local
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from astropy.io import fits
import numpy as np
from sys import exit
import pywt
import definition

def wavelet(data, levels):
    return pywt.wavedec2(data, 'haar', level=levels)


def imshow(data, norm_low, norm_high, cmap="hot", dpi=15, figsize=(50, 50)):
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(data, cmap=cmap, origin='lower', interpolation="none", norm=Normalize(norm_low, norm_high))
    plt.show()
