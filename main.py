import matplotlib.pyplot as plt
import skimage

from plotting import *
import definition
import artefacts
import artificial_cores

from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.visualization import astropy_mpl_style

from skimage.filters import gaussian, unsharp_mask, threshold_otsu, threshold_local

import numpy as np
from tqdm import tqdm
import scipy.fft as scifft
from scipy import interpolate
from scipy.ndimage import gaussian_filter

from typing import Tuple, List
import pywt

import time




# Notes:
"""
The data is not square, the top and bottom edges are spikey-ish, making auto-detection of scale tricky
A log colour scale works better than regular
Data can be slices easily as it is a numpy array, using memmap is probably best
The lab computers can load the entire image in about 3 mins
There is only one HDU (the primary one), we could add our masks to the same FITS file
"""


# Functions using this decorator can either be called directly, 
# or called via the .plot(<input>) attribute and whatever is returned is plotted simultaniously
def plottable(fig_index, **fig_kwargs):
    def plot_func(func_to_plot, *args, **kwargs):
        ret_val = func_to_plot(*args, **kwargs)
        plot_general(ret_val, fig_index=fig_index, **fig_kwargs)
        return ret_val

    def decorator(func):
        func.plot = lambda *args, **kwargs: plot_func(func, *args, **kwargs)
        return func
    return decorator


class Classifier:
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
    

        # For artifical core testing, one sublist for each cutout
        # [[[Ys], [Xs]], [[Ys], [Xs], ...]]
        self.art_cores_coords = [[]]
        self.art_artefacts_coords = [[]]


    @plottable(fig_index=1, title="Basic unsharp masking")
    def unsharp_mask_basic(img, kernel_size=1, weight=1.0, hi_threshold=None):
        gaus = gaussian(img, sigma=kernel_size)
        mask = img - gaus
        applied = img + weight * mask
        if hi_threshold:
            applied[applied >= hi_threshold] = hi_threshold
        return mask, applied


    @plottable(fig_index=2, title="Gaussian lp mask")
    def gaussian_lp_mask(radius: int, shape):
        mask = np.zeros(shape)
        center = np.array((shape[0]/2, shape[1]/2))
        for y in range(shape[0]):
            for x in range(shape[1]):
                distance_to_center = np.linalg.norm(np.array((y, x)) - center)
                mask[y,x] = np.exp(-distance_to_center**2 / (2*radius*radius))
        return mask


    @plottable(fig_index=3, title="Low pass filter using the frequency domain")
    def low_pass_filter_fourier(img, lp_filter_radius=100, weight=1, lo_threshold=0.02, hi_threshold=None):
        img_freqs = np.fft.fft2(img)

        centered_img_freqs = np.fft.fftshift(img_freqs)

        gaus_low_freq_filter_mask = Classifier.gaussian_lp_mask(radius=lp_filter_radius, shape=img_freqs.shape)

        filtered_centered_img_freqs = centered_img_freqs * gaus_low_freq_filter_mask

        filtered_img_freqs = np.fft.ifftshift(filtered_centered_img_freqs)

        filtered_img = np.fft.ifft2(filtered_img_freqs)

        # When returning, return log of the abs value of frequencies for more viewable plots
        return  np.log(1+np.abs(img_freqs)), \
                np.log(1+np.abs(centered_img_freqs)), \
                gaus_low_freq_filter_mask, \
                np.log(1+np.abs(filtered_centered_img_freqs)), \
                np.log(1+np.abs(filtered_img_freqs)), \
                np.abs(filtered_img)


    @plottable(fig_index=4, title="Wavelet")
    def discrete_2d_wavelet(data, levels):
        return pywt.wavedec2(data, 'haar', level=levels)
    
    
    def unsharp_mask_fft(img, lo_threshold_percentage=0.02):
        ...

    # Insert artificial cores to be detected, saving their positions
    def insert_artificial_cores(self, kernel_size=10, amount=1333, intensity="Random", int_min=50, int_max=350):
        for i, cutout in enumerate(self.cutouts):
            self.cutouts[i], self.art_cores_coords[i], _ = artificial_cores.insert_art_cores(cutout, kernel_size, amount, intensity=intensity, int_min=int_min, int_max=int_max)
    
    def insert_artificial_cores_new(self, size_min=5, size_max=120, amount=1333, intensity="Random", int_min=50, int_max=350):
        for i, cutout in enumerate(self.cutouts):
            self.cutouts[i], self.art_cores_coords[i], _ = artificial_cores.insert_art_cores_2(cutout, amount=amount, size_min=size_min, size_max=size_max, intensity=intensity, int_min=int_min, int_max=int_max)
            
            
    def insert_artificial_artefacts(self, amount=1333, intensity="Random", int_min=50, int_max=350):
        for i, cutout in enumerate(self.cutouts):
            self.cutouts[i], self.art_artefacts_coords[i], _ = artificial_cores.insert_art_artefacts(cutout, amount, intensity=intensity, int_min=int_min, int_max=int_max)
    
    # Get a list of coordinates of identified artefacts.
    def identify_artefacts(self) -> List[Tuple]:
        ...

    def get_mass(self, data, rows, cols, lengths):
        mass_list = []
        for i in range(len(rows)):
            if rows[i] > len(data) - lengths[i] or cols[i] > len(data[0]) - lengths[i] or cols[i] - lengths[i] < 0 or rows[i] - lengths[i] < 0:
                mass_list.append(0)
                continue
            
            circle_mask = artefacts.create_filled_circular_mask(int(lengths[i])*2, int(lengths[i])*2)
            data_square = data[(rows[i]-int(lengths[i])):(rows[i]+int(lengths[i])), (cols[i]-int(lengths[i])):(cols[i]+int(lengths[i]))]
            mass = np.sum(data_square * circle_mask)
            mass_list.append(mass)
        return mass_list
        
    def get_radius(self, data, rows, cols):
        radius_list = []
        for i in range(len(rows)):
            if rows[i] > len(data) - 30 or cols[i] > len(data[0]) - 30 or cols[i] - 30 < 0 or rows[i] - 30 < 0:
                radius_list.append(0)
                continue
            data_square = data[(rows[i]-30):(rows[i]+30), (cols[i]-30):(cols[i]+30)]
            plt.imshow(data_square)
            avers = artefacts.check_circular(data_square, 30, 30, 60, 60, 50)
            peak = avers[0]
            xs = range(0, len(avers))
            spl = interpolate.splrep(xs, avers)
            x2 = np.linspace(0, len(avers), len(avers)*100)
            y2 = interpolate.splev(x2, spl)
            for i in range(len(x2)):
                if y2[i] < peak/2:
                    radius_list.append(x2[i])
                    break
                else:
                    radius_list.append(0)
        return radius_list
            

    def run(self, unsharp_mask, wavelet, insert_artificial_cores=True, insert_artificial_artefacts=True): 
        # Definition parameters
        length = 131  # Size of box to expand around peaks when checking against the definition of a d.c.
        mult = 5  # Factor that peak has to exceed surrounding standard deviation
        lowest_peak_height = 1  # Minimum above surrounding mean value
        check_bbox_size = 3  # Size of bounding box around wavelet peaks for local maximas (This doesnt really make any sense, why would the peak not be where the wavelet identified it?)
        wavlet_levels = 3  # Number of levels to run wavelet
        wavelet_absolute_threshold = 5  # Aboslute mimimum of the summed wavlet peaks
        min_dist_between_peaks = 5  # Minimum number of pixels required between each peak
        visual_padding = 31  # Padding around indetified peaks to be shown when plotting
        
        artificial_cores = 10  # Number of artificial cores to insert
        artificial_kernel_size = 15
        intensity_value_art_cores = "Random" #Random intensity value if "Random", write number for fixed intensity
        artificial_cores_size_min = 5 #min radius
        artificial_cores_size_max = 25 #max radius
        artificial_cores_intensity_min = 50 #minimum intensity value 
        artificial_cores_intensity_max = 170 #minimum intensity value high
        
        artificial_artefacts = 10 #Number of artificial artefacts to insert
        intensity_value_art_artefacts = "Random" #Random intensity value if "Random", write number for fixed intensity
        artificial_artefacts_intensity_min = 25 #minimum intensity value artefacts
        artificial_artefacts_intensity_max = 75 #maximum intensity value artefacts
        
        unsh_mask_absolute_threshold = 5  # Aboslute mimimum of the unsharp mask
        unsh_mask_sigma = 1 # Sigma of unsharp mask
        
        if insert_artificial_cores:
            print("Inserting artificial cores")
            #self.insert_artificial_cores(amount=artificial_cores, kernel_size=artificial_kernel_size, intensity=intensity_value_art_cores, int_min=artificial_cores_intensity_min, int_max=artificial_cores_intensity_max)
            self.insert_artificial_cores_new(amount=artificial_cores, size_min=artificial_cores_size_min, size_max=artificial_cores_size_max, intensity=intensity_value_art_cores, int_min=artificial_cores_intensity_min, int_max=artificial_cores_intensity_max)
            print("Insertion done", "\n")
        if insert_artificial_artefacts:
            print("Inserting artificial artefacts")
            self.insert_artificial_artefacts(amount=artificial_artefacts, intensity=intensity_value_art_artefacts, int_min=artificial_artefacts_intensity_min, int_max=artificial_artefacts_intensity_max)
            print("Insertion done", "\n")

        for i, slice in enumerate(tqdm(self.cutouts)):
            
            plot(slice, cmap="hot", norm=colors.Normalize(0, 70))
            
            # Full pipeline using wavelet
            # (Unsharp mask > Thresholding)
            # Wavelet > Defentition check > List of cores
            
            processed_data = []
            masks = []
            
            # Calculate unsharp mask
            if unsharp_mask:
                print("\n", "Creating unsharp mask")
                start = time.time()
                unsh = (slice - gaussian(slice, sigma=unsh_mask_sigma))
                unsh_mask = unsh > unsh_mask_absolute_threshold
                processed_data.append(unsh)
                masks.append(unsh_mask)
                end = time.time()
                print("Creating unsharp mask done")
                print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(end - start)), "\n")  

            # Calculate wavelet
            if wavelet:
                # Run a 'haar' wavelet function over the data
                print("\n", "Running wavelet")
                start = time.time()
                w = Classifier.discrete_2d_wavelet(slice, wavlet_levels)
                w_sums = w[1][0] + w[1][1] + w[1][2]  # Sum the horizontal, vertical, and diagonal return values
    
                # Resize summed wavelet and calculate mask
                w_sums = skimage.transform.resize(w_sums, slice.shape)
                w_mask = w_sums > wavelet_absolute_threshold
                processed_data.append(w_sums)
                masks.append(w_mask)
                
                #plot(definition.pad_mask(w_mask, 51)*slice, cmap="hot", norm=colors.Normalize(0, 70), title="Defined dense cores", dpi=300)
                end = time.time()
                print("Running wavelet done")
                print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(end - start)), "\n")
            
            for j in range(len(processed_data)):
                # Create a mask of only the peaks
                print("Creating masks of peaks...")
                start = time.time()
                local_maxes = skimage.feature.peak_local_max(processed_data[j], min_distance=min_dist_between_peaks, labels = definition.pad_mask(masks[j], check_bbox_size))
                peaks_mask = np.zeros(processed_data[j].shape, dtype=bool)
                peaks_mask[local_maxes[:, 0], local_maxes[:, 1]] = True
                end = time.time()
                print("Creating mask of peaks done")
                print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(end - start)), "\n")
                
                # Test against definition
                print("Testing agains defenition...")
                start = time.time()
                dense_cores_mask, def_plot_arr = definition.test_def(processed_data[j], peaks_mask, length, mult, lowest_peak_height, step=2, max_diff=0.005)
                padded_dense_cores_mask = definition.pad_mask(dense_cores_mask, visual_padding)
                end = time.time()
                print("Testing agains defenition done")
                print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(end - start)), "\n")
                
                # Remove artefacts
                print("Removing artefacts...")
                start = time.time()
                lr_min_mask, lr_min_plot_arr = artefacts.lr_min(processed_data[j], dense_cores_mask, 50)
                circ_avg_min_mask, circ_avg_min_plot_arr = artefacts.circ_avg_min(processed_data[j], dense_cores_mask, 50)
                artefacts_mask = lr_min_mask | circ_avg_min_mask
                dense_cores_mask = dense_cores_mask & np.logical_not(artefacts_mask)
                padded_dense_cores_mask_no_artefacts = definition.pad_mask(dense_cores_mask, visual_padding)
                padded_artefacts_mask = definition.pad_mask(artefacts_mask, visual_padding)
                
               
                end = time.time()
                print("Removing artefacts done")
                print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(end - start)), "\n")

                dense_cores_values = processed_data[j][dense_cores_mask]
                dense_cores_coordinates = list(zip(*np.where(dense_cores_mask == True)[::-1]))
                print("Dense cores identified:", len(dense_cores_coordinates), "\n")

                # Check artificial cores
                artefacts_coordinates = list(zip(*np.where(lr_min_mask & circ_avg_min_mask)[::-1]))
                print("Artefacts identidied:", len(artefacts_coordinates), "\n")
                
                if insert_artificial_cores:
                    print("Checking found cores versus inserted cores.")
                    num_found = 0
                    for c in self.art_cores_coords[i][:, :2]:
                        
                        if tuple(c)[::-1] in dense_cores_coordinates:
                            num_found += 1
                        else:
                            distances = np.linalg.norm(dense_cores_coordinates-c[::-1], axis=1)
                            min_index = np.argmin(distances)
                            if distances[min_index] < 15:
                                num_found += 1
                    
                    print(f"Found {num_found}/{len(self.art_cores_coords[i])} inserted cores. ({num_found/len(self.art_cores_coords[i])}%)")
                    nl = "\n"
                    #print(f"Did not find the following artificial cores:{nl.join(str(c) for c in self.art_cores_coords[i])}")
                
                if insert_artificial_artefacts:
                    print("Checking found artefacts versus inserted artefacts.")
                    num_found = 0
                    for c in self.art_artefacts_coords[i][:, :2]:
                        
                        if tuple(c)[::-1] in artefacts_coordinates:
                            num_found += 1
                        else:
                            distances = np.linalg.norm(artefacts_coordinates-c[::-1], axis=1)
                            min_index = np.argmin(distances)
                            if distances[min_index] < 25: #51 size of artificial artefact
                                num_found += 1
                                
                    print(f"Found {num_found}/{len(self.art_artefacts_coords[i])} inserted cores. ({num_found/len(self.art_artefacts_coords[i])}%)")
                    nl = "\n"
                    #print(f"Did not find the following artificial artefacts:{nl.join(str(c) for c in self.art_artefacts_coords[i])}")
                
                # Calculate and plot mass to radius
                peak_rows, peak_cols, _, _, _, lengths = def_plot_arr
                radius = self.get_radius(slice, peak_rows, peak_cols)
                mass_list = self.get_mass(slice, peak_rows, peak_cols, radius)
                scatter_plot(lengths, mass_list, xlabel="radius", ylabel="mass", yscale="log", xscale='log', title="Dense cores")
                
                
                if insert_artificial_cores:
                    y = np.array(self.art_cores_coords[0][:, 0], dtype=int)
                    x = np.array(self.art_cores_coords[0][:, 1], dtype=int)
                    radius = np.array(self.art_cores_coords[0][:, 3], dtype=int)
                    
    
                    artificial_mass_list = self.get_mass(slice, y, x, radius)
                    scatter_plot(radius, artificial_mass_list, xlabel="radius", ylabel="mass", yscale="log", xscale='log', title="Artificial cores")

                # Get data to plot
                padded_dense_cores = np.where(padded_dense_cores_mask, slice, slice*0.0)
                padded_dense_cores_no_artefacts = np.where(padded_dense_cores_mask_no_artefacts, slice, slice*0.0)
                padded_artefacts = np.where(padded_artefacts_mask, slice, slice*0.0)
                
                # Plot images and graphs
                # plot_general((slice, padded_dense_cores, np.where(definition.pad_mask(not_found_mask, visual_padding), slice, 0)), title="Original, Found, Not found")
                plot(padded_dense_cores, cmap="hot", norm=colors.Normalize(0, 70), title="Defined dense cores", dpi=300)
                plot(padded_dense_cores_no_artefacts, cmap="hot", norm=colors.Normalize(0, 70), title="Defined dense cores - artefacts", dpi=300)
                plot(padded_artefacts, cmap="hot", norm=colors.Normalize(0, 70), title="Artefacts", dpi=300)
                #plot_general((slice, padded_dense_cores), title="Original, Found", norm=colors.Normalize(0, 70), dpi=100)
                plot_def_and_artefacts(processed_data[j], slice, range(0, 1000), 50, length, mult, lowest_peak_height, def_plot_arr, lr_min_plot_arr, circ_avg_min_plot_arr, onlyArtefacts=False, onlyPos=True)
                
                #artefact_rows, artefact_cols = np.where(lr_min_mask | circ_avg_min_mask)
                #for j in range(len(artefact_rows)):
                #    plot(slice[(artefact_rows[j] - 25):(artefact_rows[j] + 25), (artefact_cols[j] - 25):(artefact_cols[j] + 25)], cmap="hot")
                    
            """
            # Lowpass Example
            for r in tqdm([5, 30, 60]):
                
                lp_results = Classifier.low_pass_filter_fourier(slice, lp_filter_radius=r)
                plot_general(lp_results[0], title=f"Original, r={r}")
                plot_general(lp_results[1:], title=f"After low pass filter using ft, r={r}")
            """
            
    

if __name__ == "__main__":
    plt.style.use(astropy_mpl_style)
    
    src_path = "Q1-latest-whigal-85.fits"

    # X_LOWER, X_UPPER = 118_300, 118_900
    # Y_LOWER, Y_UPPER = 8_400, 9_000

    X_LOWER, X_UPPER = 0_000, 1_000
    Y_LOWER, Y_UPPER = 0, 1000

    sc = Classifier(src_path, [Y_LOWER, Y_UPPER, X_LOWER, X_UPPER])
    sc.run(False, True, insert_artificial_cores=True, insert_artificial_artefacts=True)
