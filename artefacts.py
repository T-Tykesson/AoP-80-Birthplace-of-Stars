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

#Checks if there is symmetry in the peaks and if there is a significant difference beteween minima and maxima
def check_for_symmetry(peaksmax, peaksmin, margin_of_error):
    center_index = None
    min_start1 = 0
    min_start2 = 1
    symmetry = False
    
    for i in range(len(peaksmax)):
        if peaksmax[i] == 0:
            center_index = i
            max_left = i-1
            max_right = i + 1
    
    if center_index == None:
        print("Center index = null")
        return False
    elif  max_left < 0 or max_right > len(peaksmax):
        print("Lacking minima or maxima on sides")
    
    for i in range(len(peaksmin)):
        if peaksmin[i] > 0 and peaksmin[min_start2] < 0:
            min_start1 = i
            break
        min_start2 += 1
        
    if absolute_symmetry(peaksmax[max_left], peaksmax[max_right], margin_of_error) and absolute_symmetry(peaksmin[min_start1], peaksmin[min_start2], margin_of_error):
        if significance_of_peak(peaksmax[max_left], min_start1, minimum_peak_diff) and significance_of_peak(peaksmax[max_right] , min_start2, minimum_peak_diff):
            symmetry = True
    else:
            symmetry = False
    
    return symmetry

def significance_of_peak(local_max, local_min, threshold): #Checks if given values are significantly different
    if datap[local_max] - datap[local_min] > threshold:
        return True
    else:
        return False

minimum_peak_diff = 15
strictness = 1

full_data = False #Ändra till false om man inte vill dela upp hela datan
file_path = "Q1-latest-whigal-85.fits"

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
    x_low = 11540*5
    x_high = 11565*5
    y_low = 636*5
    y_high = 656*5
    data = data[y_low:y_high, x_low:x_high]
    

maxim = data.argmax()
index = np.unravel_index(maxim, data.shape)
#print(index)
plotting.plot_figure(data,"Artefakt")

size = 40

datap = data[index[0], (index[1]-size):(index[1]+size)]
X = np.linspace(0,2*size-1, num=2*size)

peaksh, _ = find_peaks(datap) #List of local maxima
peaksl, _ = find_peaks(-datap) #List of local minima


print(size-peaksl)
print(size-peaksh)
print(check_for_symmetry((size-peaksh), (size-peaksl), strictness))

#plt.plot(peaksh, datap[peaksh], "x")
#plt.plot(peaksl, datap[peaksl], 'x')
#plt.plot(X,datap)
#plt.show()


## Creates a ring with radius=radius with center at center. h = height of data and w = width of data
def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask_inner = dist_from_center <= (radius-1)
    mask_outer = dist_from_center <= radius
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

def distance_minima(data, center, h, w, radius_max):
    aver = check_circular(data, center, h, w, radius_max)
    func = (range(len(aver)),aver)
    peak, _ = find_peaks(-func, width=(0,20))
    return peak


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


def find_artefacts_from_coords(data, coords, width, radius_max=40):
    #coords is y,x coordinates tuples
    for i in range(len(coords)):
        y, x = coords[i]
        center = y, x
        y_low = y - width
        y_high = y + width
        x_low = x - width
        y_high = y + width
        h = y_high - y_low   
        w = x_high - x_low
        
        data_slice = data[y_low:y_high, x_low:x_high]
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

#%%




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
