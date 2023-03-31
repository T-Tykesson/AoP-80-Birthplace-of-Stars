import plotting
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from tqdm import tqdm

def absolute_symmetry(value1, value2):
    if abs(value1) == abs(value2):
        return True
    else:
        return False

def check_for_symmetry(peaksmax, peaksmin, margin_of_error):
    center_index = None
    min_start1 = 0
    min_start2 = 1
    symmetry = False
    for i in peaksmax:
        if i == 0:
            center_index = i
    
    if center_index == None:
        print("Center index = null")
        return False
    
    for i in range(len.peaksmin):
        if peaksmin[i] > 0 and peaksmin[min_start2] < 0:
            min_start1 = i
            exit
        min_start2 += 1
    if peaksmin[center_index+1] == peaksmin[center_index-1] and peaksmax[min_start1] ==peaksmax[min_start1]:
        return True
    

full_data = False #Ändra till false om man inte vill dela upp hela datan
file_path = "C:/Users/joaki/Pictures/Q1-latest-whigal-85.fits"

if full_data:
    data_list = []
    xslice = 10000
    yslice = 7000

    with fits.open(file_path, use_fsspec=True, fsspec_kwargs={"anon": True}) as hdul:  
       #cutout = hdul[0].section[0:1750, 0:2500] 
        for i in tqdm(range(0, int(7000/yslice))):
            for j in tqdm(range(0, int(120000/xslice))):
                cutout = hdul[0].section[yslice*i:yslice*i + yslice, xslice*j:xslice*j+xslice]
                data_list.append(cutout)
else:
    data = fits.getdata(file_path)
    x_low = 11840*5
    x_high = 11875*5
    y_low = 725*5
    y_high = 755*5
    data = data[y_low:y_high, x_low:x_high]
    

maxim = data.argmax()
index = np.unravel_index(maxim, data.shape)
print(index)
plotting.plot_figure(data,"Artefakt")

size = 40

datap = data[index[0], (index[1]-size):(index[1]+size)]
X = np.linspace(0,2*size-1, num=2*size)

peaksh, _ = find_peaks(datap) #List of local maxima
peaksl, _ = find_peaks(-datap) #List of local minima


print(size-peaksl)
print(size-peaksh)

plt.plot(peaksh, datap[peaksh], "x")
plt.plot(peaksl, datap[peaksl], 'x')
plt.plot(X,datap)
plt.show()


'''
    x_low = 15470*5
    x_high = 15495*5
    y_low = 575*5
    y_high = 595*5

    x_low = 11540*5
    x_high = 11565*5
    y_low = 636*5
    y_high = 656*5
    
    x_low = 11840*5
    x_high = 11875*5
    y_low = 725*5
    y_high = 755*5
'''
