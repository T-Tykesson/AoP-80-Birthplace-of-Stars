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

"""
def imshow(data, norm_low, norm_high, cmap="hot", dpi=15, figsize=(50, 50)):
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(data, cmap=cmap, origin='lower', interpolation="none", norm=Normalize(norm_low, norm_high))
    plt.show()

file_path = ""
data = fits.getdata(file_path)
x_low = 0
x_high = 20000
y_low = 0
y_high = 20000
data = data[y_low:y_high, x_low:x_high]

w = wavelet(data, 3)

h = w[1][0]
v = w[1][1]
d = w[1][2]

threshold = 15
length = 201
mult = 15
lowest_val = 50

imshow(data, 0, 70, dpi=100)

imshow(h, 0, 30)
imshow(v, 0, 30)
imshow(d, 0, 20)

h = h > threshold
v = v > threshold
d = d > threshold

h = definition.pad_mask(h, 21)
v = definition.pad_mask(v, 21)

binary = np.bitwise_and(h, v)
binary = np.bitwise_and(binary, d)

binary = definition.mask_to_data(data, binary)
binary_pad = definition.pad_mask(binary, 201)
imshow(binary_pad * data, 0, 70)

imshow(definition.pad_mask(definition.test_on_def(data, binary, length, mult, lowest_val), 201) * data, 0, 70)
"""
