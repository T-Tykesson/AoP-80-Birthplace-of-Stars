# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 16:39:28 2023

@author: Tage


data split
"""
from astropy.io import fits
from tqdm import tqdm

def get_splitted_data(file_path, yslice, xslice):
    data_list = []
    with fits.open(file_path, memmap=True) as hdul:  
       #cutout = hdul[0].section[0:1750, 0:2500] 
        for i in tqdm(range(0, int(7000/yslice))):
            for j in tqdm(range(0, int(120000/xslice))):
                cutout = hdul[0].section[yslice*i:yslice*i + yslice, xslice*j:xslice*j+xslice]
                data_list.append(cutout)
    return data_list
import numpy as np
def get_data_slice(file_path, ylow, yhigh, xlow, xhigh):
    data = fits.getdata(file_path)
    data = data[ylow:yhigh, xlow:xhigh]
    return data

          