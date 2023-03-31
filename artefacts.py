import plotting
from astropy.io import fits
import numpy as np
import get_data
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from tqdm import tqdm

def absolute_symmetry(value1, value2, margin):
    if abs(value1) == abs(value2) or abs(value1) == abs(value2) + margin or abs(value1) == abs(value2) - margin:
        return True
    else:
        return False

def check_for_symmetry(peaksmax, peaksmin, margin_of_error):
    center_index = None
    min_start1 = 0
    min_start2 = 1
    symmetry = False
    
    for i in range(len(peaksmax)):
        if peaksmax[i] == 0:
            center_index = i
    
    if center_index == None:
        print("Center index = null")
        return False
    
    for i in range(len(peaksmin)):
        if peaksmin[i] > 0 and peaksmin[min_start2] < 0:
            min_start1 = i
            break
        min_start2 += 1
        
    if absolute_symmetry(peaksmax[center_index - 1], peaksmax[center_index + 1], margin_of_error) and absolute_symmetry(peaksmin[min_start1], peaksmin[min_start2], margin_of_error):
        symmetry = True
    
    return symmetry
## Creates a ring with radius=radius with center at center. h = height of data and w = width of data
def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    maski = dist_from_center <= (radius-1)
    masko = dist_from_center <= radius
    mask = masko^maski
    return mask

## Takes the average intensity of the ring at radius r. Center is the center of the dense core, h = height of data and w = width of data, radius_max is the total size of the circle.
def check_circular(data, center, h, w, radius_max):
    aver = np.zeros(radius_max)
    aver[0] = data[center[1], center[0]]
    for r in range(1,radius_max):
        mask = create_circular_mask(h,w, center = center, radius = r)
        aver[r] = np.sum(mask*data)/(mask > 0).sum()
    return aver

## Plots the average intensity at radius r. Center is the center of the dense core, h = height of data, w = width of data, radius_max is the total size of the circle.
def plot_intensity2radius(data, center, h, w, radius_max):   
    aver = check_circular(data, center, h, w, radius_max)
    plt.plot(range(len(aver)), aver)
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

def find_artefacts(data, y_low, y_high, x_low, x_high, radius_max=40):
    data_slice = data[y_low:y_high, x_low:x_high]
    #print(index)
    maxim = data_slice.argmax()
    index = np.unravel_index(maxim, data_slice.shape)
    h = y_high - y_low   
    w = x_high - x_low
    center = (index[1], index[0])
    plotting.plot_figure(data_slice,"Artefakt")
    plot_intensity2radius(data_slice, center, h, w, radius_max)



"Test"

file_path = "C:/Users/Tage/Programmering/AoP80/Q1-latest-whigal-85.fits"
x_low = 15470*5
x_high = 15495*5
y_low = 575*5
y_high = 595*5
data = get_data.get_data_slice(file_path, 0, 10000, 0, 80000)

find_artefacts(data, y_low, y_high, x_low, x_high)



#plt.plot(peaksh, datap[peaksh], "x")
#plt.plot(peaksl, datap[peaksl], 'x')
#plt.plot(X,datap)
#plt.show()

