# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 15:21:27 2023

@author: Tage
"""

from skimage.filters import gaussian, unsharp_mask

def gaussian_threshold(data, sigma, offset):
    g = gaussian(data, sigma=sigma)
    binary = data > (g + offset)
    return g, binary
