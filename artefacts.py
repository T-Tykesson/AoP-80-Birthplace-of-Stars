import plotting
from astropy.io import fits
import numpy as np
import get_data
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from tqdm import tqdm
from definition import create_box_around_peaks
from scipy import interpolate

minimum_peak_diff = 15
strictness = 1
size = 40
def check_symmetry(orgdata, centre_coords):
    ymid = centre_coords[0]
    xmid = centre_coords[1]
    

    datap = orgdata[xmid, (ymid-size):(ymid+size)]
    X = np.linspace(0,2*size-1, num=2*size)

    peaksh, _ = find_peaks(datap) #List of local maxima
    peaksl, _ = find_peaks(-datap) #List of local minima
    print(size-peaksl)
    print(size-peaksh)
    symmetry_y = symmetry_2d(size - peaksh, size - peaksl, strictness, datap, xmid)
    print("Likely an artefact:")
    print(symmetry_y)
    plt.plot(peaksh, datap[peaksh], "x")
    plt.plot(peaksl, datap[peaksl], 'x')
    plt.plot(X,datap)
    plt.show()
    
    datap = orgdata[ymid, (xmid-size):(xmid+size)]
    X = np.linspace(0,2*size-1, num=2*size)

    peaksh, _ = find_peaks(datap) #List of local maxima
    peaksl, _ = find_peaks(-datap) #List of local minima
    print(size-peaksl)
    print(size-peaksh)
    symmetry_y = symmetry_2d(size - peaksh, size - peaksl, strictness, datap, ymid)
    print("Likely an artefact:")
    print(symmetry_y)
    plt.plot(peaksh, datap[peaksh], "x")
    plt.plot(peaksl, datap[peaksl], 'x')
    plt.plot(X,datap)
    plt.show()
    
    
    
#Checks if there is symmetry in the peaks and if there is a significant difference beteween minima and maxima
def symmetry_2d(peaksmax, peaksmin, margin_of_error, datam, c):
    max_right = None
    max_left = None
    min_right = None
    min_left = None
    symmetry = False
    
    for i in range(len(peaksmax)):
        if peaksmax[i] <= -10:
            max_right = peaksmax[i]
            print(max_right)
            for j in reversed(range(i)):
                if peaksmax[j] > 10:
                    max_left = peaksmax[j]
                    print(max_left)
                    break
            break
        
    
    if  max_left == None or max_right == None:
        print("Lacking maxima on sides")
        return False
    
    for i in range(len(peaksmin)):
        if peaksmin[i] <= -10:
            min_right = peaksmin[i]
            print(min_right)
            for j in reversed(range(i)):
                if peaksmin[j] > 10:
                    min_left = peaksmin[j]
                    print(min_left)
                    break
            break
    
    if  min_left == None or min_right == None:
        print("Lacking minima on sides")
        return False
        
    if absolute_symmetry(max_left, max_right, margin_of_error) and absolute_symmetry(min_left, min_right, margin_of_error):
        if significance_of_peak(max_right , min_right, minimum_peak_diff, datam) and significance_of_peak(max_left, min_left, minimum_peak_diff, datam):
            symmetry = True
    else:
            symmetry = False
            print("Symmetryfail")
    
    return symmetry

def significance_of_peak(local_max, local_min, threshold, datag): #Checks if given values are significantly different

    if datag[local_max + size] - datag[local_min + size] > threshold:
        return True
    else:
        print("significance fail")
        return False
    
    

def absolute_symmetry(value1, value2, margin):
    if abs(value1) == abs(value2) or abs(value1) == abs(value2) + margin or abs(value1) == abs(value2) - margin:
        return True
    else:
        return False




#minimum_peak_diff = 15
#strictness = 1

#full_data = False #Ändra till false om man inte vill dela upp hela datan
#file_path = "Q1-latest-whigal-85.fits"

#if full_data:
#    data_list = []
#    xslice = 10000
#    yslice = 7000

#    with fits.open(file_path, use_fsspec=True, fsspec_kwargs={"anon": True}) as hdul:  
#       #cutout = hdul[0].section[0:1750, 0:2500] 
#        for i in tqdm(range(0, int(7000/yslice))):
#            for j in tqdm(range(0, int(120000/xslice))):
#                cutout = hdul[0].section[yslice*i:yslice*i + yslice, xslice*j:xslice*j+xslice]
#                data_list.append(cutout)
#else:
#    data = fits.getdata(file_path)
#    x_low = 11540*5
#    x_high = 11565*5
#    y_low = 636*5
#    y_high = 656*5
#    data = data[y_low:y_high, x_low:x_high]
    

#maxim = data.argmax()
#index = np.unravel_index(maxim, data.shape)
#print(index)
#plotting.plot_figure(data,"Artefakt")

#size = 40

#datap = data[index[0], (index[1]-size):(index[1]+size)]
#X = np.linspace(0,2*size-1, num=2*size)

#peaksh, _ = find_peaks(datap) #List of local maxima
#peaksl, _ = find_peaks(-datap) #List of local minima


#print(size-peaksl)
#print(size-peaksh)
#print(check_for_symmetry((size-peaksh), (size-peaksl), strictness))

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
    mask = mask_outer^mask_inner
    return mask

# Calculates average intensity of rings from radius 0 to given max radius at every 1 in the given mask.
def check_circular_multiple(data, mask, radius_max):
    matrices_coords = create_box_around_peaks(mask, radius_max*2 + 1)
    matrices = data[matrices_coords]
    avers = np.zeros((len(matrices), radius_max))
    avers[:,0] = matrices[:, radius_max, radius_max]
    for r in range(1, radius_max):
        #print(r)
        circ_mask = create_circular_mask(radius_max*2 + 1, radius_max*2 + 1, radius=r)
        avers[:, r] = np.sum(matrices * circ_mask, axis=(1,2)) / (circ_mask > 0).sum()
    return avers

# Checks the first local minima in the average intensity of rings graph, at every 1 in the given mask.
# If its x is between low and high it is set as an artefact.
# Returns a mask where at every 1 is a predicted artefact, also returns
# an array for plotting.
def circ_avg_min(data, mask, radius_max, low=9, high=12, s=1):
    avers = check_circular_multiple(data, mask, radius_max)
    artefact_mask = np.zeros(mask.shape, dtype=bool)
    center_ys, center_xs = np.where(mask)
    
    artefact_list = []
    smoothed_list = []
    smoothed_index_list = []
    
    for i in range(len(avers)):
        center_x = center_xs[i]
        center_y = center_ys[i]
        
        xs = range(0, len(avers[i]))
        spl = interpolate.splrep(xs, avers[i], s=s)
        ynew = interpolate.BSpline(*spl)(xs)
        
        mins, _ = find_peaks(-avers[i])
        mins_smooth, _ = find_peaks(-ynew)
        first_right_index = mins[0] if len(mins) > 0 else None
        first_right_index_smooth = mins_smooth[0] if len(mins_smooth) > 0 else None
        
        artefact = False
        if first_right_index != None:
            if low <= first_right_index <= high:
                artefact = True
                artefact_mask[center_y, center_x] = True
        if first_right_index_smooth != None:
            if low <= first_right_index_smooth <= high:
                artefact = True
                artefact_mask[center_y, center_x] = True
        
        smoothed_list.append(ynew)
        artefact_list.append(artefact)
        smoothed_index_list.append(first_right_index_smooth)
        
    
    return artefact_mask, [center_xs, center_ys, smoothed_list, smoothed_index_list, avers, artefact_list]

# Checks the first local minima to the left and right of the center point, at every 1 in the given mask.
# If its x is between low and high it is set as an artefact.
# Returns a mask where at every 1 is a predicted artefact, also returns
# an array for plotting.
def lr_min(data, mask, x_view, low = 8, high = 20, s=15):
    artefact_mask = np.zeros(mask.shape, dtype=bool)
    center_ys, center_xs = np.where(mask)
    center_xs_pad = np.expand_dims(center_xs, 1)
    center_xs_pad = np.pad(center_xs_pad, ([0, 0], [x_view*2, 0]), 'edge')
    offsets = np.pad(np.array([range(-x_view, x_view + 1)]), ([0, len(center_xs_pad) - 1], [0, 0]), 'edge')
    center_xs_pad = center_xs_pad + offsets
    center_xs_pad = np.maximum(center_xs_pad, 0)
    center_xs_pad = np.minimum(center_xs_pad, len(data[0]) - 1)
    values_list = data[np.expand_dims(center_ys,1), center_xs_pad]
    
    smoothed_list = []
    first_left_index_list = []
    first_right_index_list = []
    artefact_list = []
    
    for i in range(len(center_ys)):
        center_x = center_xs[i]
        center_y = center_ys[i]
        values = values_list[i]
        
        xs = range(-x_view, x_view + 1)
        spl = interpolate.splrep(xs, values, s=s)
        ynew = interpolate.BSpline(*spl)(xs)
        
        mins, _ = find_peaks(-ynew)
        mins_x_from_center = mins - x_view
        
        left = mins[mins_x_from_center < 0]
        right = mins[mins_x_from_center > 0]
        
        first_left_index = left[-1] if len(left) > 0 else None
        first_right_index = right[0] if len(right) > 0 else None
        
        artefact = False
        if first_left_index != None and first_right_index != None:
            if -high <= first_left_index - x_view - 1 <= -low and low <= first_right_index - x_view - 1 <= high:
                artefact = True
                artefact_mask[center_y, center_x] = True
        
        smoothed_list.append(ynew)
        first_left_index_list.append(first_left_index)
        first_right_index_list.append(first_right_index)
        artefact_list.append(artefact)
        
    return artefact_mask, [center_xs, center_ys, smoothed_list, first_left_index_list, first_right_index_list, artefact_list]

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
    peak, _ = find_peaks(-aver, width=(0,20))
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
    print(distance_minima(data_slice, center, h, w, radius_max))
    


def find_artefacts_from_coords(data, coords, width, radius_max=40):
    #coords is y,x coordinates tuples
    for i in range(len(coords)):
        y, x = coords[i]
        #center = y, x
        y_low = max(y - width, 0)
        print("ylow:", y_low)
        y_high = min(y + width, 7000)
        print("yhigh:", y_high)
        x_low = max(x - width, 0)
        x_high = min(x + width, 120000)
        h = y_high - y_low   
        print("h:", h)
        w = x_high - x_low
        print("w:", w)
        
        center = (h//2, w//2)
        print(x, y)
        data_slice = data[y_low:y_high, x_low:x_high]
        plotting.plot_figure(data_slice,f"Artefact? ({x},{y})")
        plot_intensity2radius(data_slice, center, h, w, radius_max)


"Test"

#file_path = "Q1-latest-whigal-85.fits"
#x_low = 15470*5
#x_high = 15495*5
#y_low = 575*5
#y_high = 595*5
#data = get_data.get_data_slice(file_path, 0, 10000, 0, 80000)

#find_artefacts(data, y_low, y_high, x_low, x_high)

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
    
    check_symmetry(data_slice, center)



"Test"

#file_path = ""
#x_low = 15470*5
#x_high = 15495*5
#y_low = 575*5
#y_high = 595*5
#data = get_data.get_data_slice(file_path, 0, 10000, 0, 80000)

#find_artefacts(data, y_low, y_high, x_low, x_high)



#plt.plot(peaksh, datap[peaksh], "x")
#plt.plot(peaksl, datap[peaksl], 'x')
#plt.plot(X,datap)
#plt.show()

