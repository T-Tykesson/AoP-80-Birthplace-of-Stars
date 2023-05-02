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
from pathlib import Path




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
        
        # For artifical core testing, one sublist for each cutout
        # [[[Ys], [Xs]], [[Ys], [Xs], ...]]
        self.art_cores_coords = []
        self.art_artefacts_coords = []
        
        self.limits = limits
        self.single_slice = single_slice

        with fits.open(promise_path, memmap=True) as fits_data:
            
            ymax, xmax = fits_data["PRIMARY"].data.shape

            if not single_slice:

                yslice_size = limits[1]
                xslice_size = limits[3]

                for i in tqdm(range(0, int(ymax/yslice_size))):
                    for j in tqdm(range(0, int(xmax/xslice_size))):
                        self.cutouts.append(fits_data["PRIMARY"].data[yslice_size*i : (yslice_size*i+yslice_size), xslice_size*j : (xslice_size*j+xslice_size)])
                        self.art_cores_coords.append([])
                        self.art_artefacts_coords.append([])
            else:
                self.cutouts.append(fits_data["PRIMARY"].data[ limits[0]:limits[1], limits[2]:limits[3] ])
                self.art_cores_coords.append([])
                self.art_artefacts_coords.append([])


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
        
        m = artificial_cores.create_gaussian_filter(radius)
        m = np.pad(m, ([(shape[0]-radius)//2, (shape[0]-radius)//2],[(shape[1]-radius)//2, (shape[1]-radius)//2]))
        return m


    @plottable(fig_index=3, title="Low pass filter using the frequency domain")
    def low_pass_filter_fourier(self, img, lp_filter_radius=100, weight=1, lo_threshold=0.02, hi_threshold=None):
        img_freqs = np.fft.fft2(img)

        centered_img_freqs = np.fft.fftshift(img_freqs)

        gaus_low_freq_filter_mask = Classifier.gaussian_lp_mask(radius=lp_filter_radius, shape=img_freqs.shape)

        filtered_centered_img_freqs = centered_img_freqs * gaus_low_freq_filter_mask

        filtered_img_freqs = np.fft.ifftshift(filtered_centered_img_freqs)

        filtered_img = np.fft.ifft2(filtered_img_freqs)

        # When returning, return log of the abs value of frequencies for more viewable plots
        return  np.abs(filtered_img)
                #np.log(1+np.abs(img_freqs)), \
                #np.log(1+np.abs(centered_img_freqs)), \
                #gaus_low_freq_filter_mask, \
                #np.log(1+np.abs(filtered_centered_img_freqs)), \
                #np.log(1+np.abs(filtered_img_freqs))
            

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
            if lengths[i] == 0:
                mass_list.append(0)
            
            else:
                circle_mask = artefacts.create_filled_circular_mask(int(lengths[i])*2, int(lengths[i])*2)
                data_square = data[(rows[i]-int(lengths[i])):(rows[i]+int(lengths[i])), (cols[i]-int(lengths[i])):(cols[i]+int(lengths[i]))]
                mass = np.sum(data_square * circle_mask)
                mass_list.append(mass)
                
        return mass_list
        
    def get_radius(self, data, rows, cols):
        radius_list = []
        size = 25
        for i in range(len(rows)):
            if rows[i] > len(data) - (size+1) or cols[i] > len(data[0]) - (size+1) or cols[i] - (size+1) < 0 or rows[i] - (size+1) < 0:
                #raise ValueError("Invalid input.")
                radius_list.append(0)
                continue
            
            data_square = data[(rows[i]-(size)):(rows[i]+size+1), (cols[i]-(size)):(cols[i]+size+1)]
            
            # Get averages of a circle of values expanding from the peak,
            # to a radius of size. avers is a list of averages indexed by radial pixels from the peak
            avers = artefacts.check_circular(data_square, size, size, 2*size+1, 2*size+1, size-5)
            peak = avers[0]
            base_line = np.min(avers)
            
            # Approxomate a smooth curve of these averages using smooth spline approximation
            #spl = interpolate.splrep(range(len(avers)), avers)
            
            # Smooth this curve further by interpolating the values by a factor of 100
            x2 = range(len(avers))
            y2 = avers #interpolate.splev(x2, spl)
            
            
            
            # Traverse through the averages finding the first average to <= half the peak value
            # ( Full width half max )
            found_radius = False
            for j in x2:
                if (y2[j] - base_line) <= (peak - base_line)/2:
                    radius = x2[j]
                    radius_list.append(radius)
                    found_radius = True
                    break
                
            if not found_radius:
                radius = 0
                radius_list.append(0)

        return radius_list
            #if radius > 20:
            #    plt.title(f"Plotting dense core if radius greater than 20, radius: {radius}")
            #    plt.imshow(data_square)
            #    plt.show()


    def plot_cores_from_catalog(self, catalog):
        k = 0
        found_cores = []
        for i in catalog[0]: #creating a list of pictures of dense cores
            y=i[0]
            x=i[1]
            size = 51
            data_square = self.cutouts[0][(y-size):(y+size), (x-size):(x+size)]
            found_cores.append(data_square)
            if k > 50:
                break
            k += 1
        
        # Plotting dense cores
        fig = plt.figure(figsize=(26, 10))
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0.2)
        columns = 8
        rows = 3
        
        # ax enables access to manipulate each of subplots
        ax = []
        for i in range(columns*rows):
            # create subplot and append to ax
            ax.append(fig.add_subplot(rows, columns, i+1))
            plt.grid(False)
            plt.axis('off')
            
            #ax[-1].set_title("ax:"+str(i))  # set title
            ax[-1].set_title(f"{catalog[0][i]}")  # set title as coordinates
            plt.imshow(found_cores[i], alpha=1)
    
        plt.show()  

    def check_overlap(self, catalog_1, catalog_2, pixel_perfect=False, plot=False):
        if pixel_perfect:
            overlap = np.intersect1d(catalog_1[0], catalog_2[0])
            print("Overlap:", len(overlap))
        else:
            #If 
            overlap = []
            error = 3 #Value in pixels
            for c in catalog_1[0]:
                distances = np.linalg.norm(np.array([*catalog_2[0]])-list(c), axis=1)
                min_index = np.argmin(distances)
                if distances[min_index] <= error:
                    overlap.append(c)
                    
            print("Overlap:", len(overlap))
        
        if plot:
            for i in overlap:
                y=i[0]
                x=i[1]
                size = 101
                data_square = self.cutouts[0][(y-size):(y+size+1), (x-size):(x+size+1)]
                #plt.title(f"Radius:{catalog[1][k]}, Mass:{catalog[2][k]}, Artefact:{catalog[3][k]}")
                plt.grid(False)
                plt.imshow(data_square)
                plt.show()
                #k += 1
                
    def check_in_1_not_2(self, catalog_1, catalog_2, pixel_perfect=False, plot=False, name=None):
        if pixel_perfect:
            one_not_two = np.setdiff1d(catalog_1[0],catalog_2[0])
            print(f"Only in {name}:", len(one_not_two))
        else:
            error = 3
            one_not_two = []
            for c in catalog_1[0]:
                distances = np.linalg.norm(np.array([*catalog_2[0]])-list(c), axis=1)
                min_index = np.argmin(distances)
                if distances[min_index] > error:
                    one_not_two.append(c)
                    
            print(f"Only in {name}:", len(one_not_two))
        
        if plot:
            for i in one_not_two:
                y=i[0]
                x=i[1]
                size = 51
                data_square = self.cutouts[0][(y-size):(y+size), (x-size):(x+size)]
                
                #plt.title(f"Radius:{catalog[1][k]}, Mass:{catalog[2][k]}, Artefact:{catalog[3][k]}")
                plt.grid(False)
                plt.imshow(data_square)
                plt.show()
                #k += 1
                
    def merge_catalog(self, catalog_1, catalog_2):
        catalog_1[1] = np.hstack((catalog_1[1], catalog_2[1]))
        return catalog_1
    
    def get_threshold_plot_values(self, processed_data, threshold_min, threshold_max, step, check_bbox_size, min_dist_between_peaks, length, mult, lowest_peak_height):
        xs = []
        ys = []
        print("Getting threshold values for plotting...")
        for threshold in np.arange(threshold_max, threshold_min - step, -step):
            mask = processed_data > threshold
            local_maxes = skimage.feature.peak_local_max(processed_data, min_distance=min_dist_between_peaks, labels = definition.pad_mask(mask, check_bbox_size))
            peaks_mask = np.zeros(processed_data.shape, dtype=bool)
            peaks_mask[local_maxes[:, 0], local_maxes[:, 1]] = True
            dense_cores_mask, def_plot_arr = definition.test_def(processed_data, peaks_mask, length, mult, lowest_peak_height, step=2, max_diff=0.005)
            
            xs.append(len(np.where(peaks_mask)[0]))
            ys.append(len(np.where(dense_cores_mask)[0]))
            print("Got: ", xs[-1], " ", ys[-1], "with threshold: ", threshold)
        print("Got threshold values.")
        return xs, ys
        

    def run(self, unsharp_mask, wavelet, fourier, plot_images=False, insert_artificial_cores=True, insert_artificial_artefacts=True, save=False, compare=False, merge=False, plot_threshold=False, save_plots_and_images=False): 
        # Definition parameters
        length = 61  # Size of box to expand around peaks when checking against the definition of a d.c.
        mult = 5 # Factor that peak has to exceed surrounding standard deviation
        lowest_peak_height = 0  # Minimum above surrounding mean value
        check_bbox_size = 3  # Size of bounding box around wavelet peaks for local maximas (This doesnt really make any sense, why would the peak not be where the wavelet identified it?)
        wavlet_levels = 1  # Number of levels to run wavelet
        wavelet_absolute_threshold = 0.5 # Aboslute mimimum of the summed wavlet peaks
        min_dist_between_peaks = 5  # Minimum number of pixels required between each peak
        visual_padding = 51  # Padding around indetified peaks to be shown when plotting
        
        unsh_mask_absolute_threshold = 0.1  # Aboslute mimimum of the unsharp mask
        unsh_mask_sigma = 1 # Sigma of unsharp mask
        
        artificial_cores = 1000  # Number of artificial cores to insert
        artificial_kernel_size = 15
        intensity_value_art_cores = "Random" #Random intensity value if "Random", write number for fixed intensity
        artificial_cores_size_min = 5 #min kernel size, note this is not the same as the caluclated radius using fwhm
        artificial_cores_size_max = 25 #max kernel size, note this is not the same as the caluclated radius using fwhm
        artificial_cores_intensity_min = 5 #minimum intensity value  
        artificial_cores_intensity_max = 30 #minimum intensity value 
        
        artificial_artefacts = 500 #Number of artificial artefacts to insert
        intensity_value_art_artefacts = "Random" #Random intensity value if "Random", write number for fixed intensity
        artificial_artefacts_intensity_min = 25 #minimum intensity value artefacts
        artificial_artefacts_intensity_max = 75 #maximum intensity value artefacts

        fourier_lp_filter_radius = 500 # Radius of fourier low pass filter
        fourier_absolute_threshold = 0.01 # Absolute threshold of fourier low pass filter
        
        
        if compare:
            catalog_1 = np.load("test.npy", allow_pickle=True)
            catalog_2 = np.load("test2.npy",allow_pickle=True)
            self.plot_cores_from_catalog(catalog_1[1])
            self.plot_cores_from_catalog(catalog_2[1])
            #self.plot_cores_from_catalog(catalog_1[1])
            #print(catalog_1[1][0])
            #print(catalog_2[1][0])
            print("Found in 1:", len(catalog_1[1][1]))
            print("Found in 2:", len(catalog_2[1][1]))
            self.check_overlap(catalog_1[1], catalog_2[1], pixel_perfect=False, plot=False)
            self.check_in_1_not_2(catalog_1[1], catalog_2[1], pixel_perfect=False, plot=False, name="catalog 1")
            self.check_in_1_not_2(catalog_2[1], catalog_1[1], pixel_perfect=False, plot=False, name="catalog 2")
            return None
        
        if merge:
            # First element of catalog is info, info is perserved in merge
            catalog_1 = np.load("test.npy", allow_pickle=True)
            catalog_2 = np.load("test2.npy",allow_pickle=True)
            #catalog_3 = np.load("test3.npy",allow_pickle=True)
            #catalog_4 = np.load("test4.npy",allow_pickle=True)
            merged = self.merge_catalog(catalog_1, catalog_2)
            #merged_2 = self.merge_catalog(catalog_3, catalog_4)
            #merged_tot = self.merge_catalog(merged, catalog_4)
            
            print("Saving merge")
            file_name = "Merged catalog" #Name of merged catalog
            np.save(file_name, merged)
            print("Saved merge")
            return None
            
        if insert_artificial_cores:
            print("Inserting artificial cores")
            #self.insert_artificial_cores(amount=artificial_cores, kernel_size=artificial_kernel_size, intensity=intensity_value_art_cores, int_min=artificial_cores_intensity_min, int_max=artificial_cores_intensity_max)
            self.insert_artificial_cores_new(amount=artificial_cores, size_min=artificial_cores_size_min, size_max=artificial_cores_size_max, intensity=intensity_value_art_cores, int_min=artificial_cores_intensity_min, int_max=artificial_cores_intensity_max)
            print("Insertion done", "\n")
        if insert_artificial_artefacts:
            print("Inserting artificial artefacts")
            self.insert_artificial_artefacts(amount=artificial_artefacts, intensity=intensity_value_art_artefacts, int_min=artificial_artefacts_intensity_min, int_max=artificial_artefacts_intensity_max)
            print("Insertion done", "\n")
        current_slice = 0
        
        for i, slice in enumerate(tqdm(self.cutouts)):
            plot(slice, cmap="hot", norm=colors.Normalize(0, 70))
            
            if not self.single_slice:
                self.limits = [Y_LOWER, Y_UPPER, X_UPPER*current_slice, X_UPPER*(current_slice+1)]
            
            # Full pipeline using wavelet
            # (Unsharp mask > Thresholding)
            # Wavelet > Defentition check > List of cores
            
            processed_data = []
            masks = []
            plot_graphs_and_images_paths = []
            
            # Calculate unsharp mask
            if unsharp_mask:
                print("\n", "Creating unsharp mask")
                start = time.time()
                unsh = (slice - gaussian(slice, sigma=unsh_mask_sigma))
                unsh_mask = unsh > unsh_mask_absolute_threshold
                processed_data.append(unsh)
                masks.append(unsh_mask)
                end = time.time()
                
                parameter_info = f"Unsharp mask, absolute threshold = {unsh_mask_absolute_threshold}, sigma = {unsh_mask_sigma}, Other parameters: length={length}, mult={mult}, lowest_peak_height={lowest_peak_height}, check_bbox_size={check_bbox_size}, min_dist_between_peaks={min_dist_between_peaks}" 
                plot_graphs_and_images_paths.append(catalog_folder_path + "unsharp_mask/" + str(mult) + "/" + str(wavelet_absolute_threshold) + "/")
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
                
                parameter_info = f"Wavelet, wavlet_levels = {wavlet_levels}, wavelet_absolute_threshold = {wavelet_absolute_threshold}, Other parameters: length={length}, mult={mult}, lowest_peak_height={lowest_peak_height}, check_bbox_size={check_bbox_size }, min_dist_between_peaks={min_dist_between_peaks}" 
                plot_graphs_and_images_paths.append(catalog_folder_path + "wavelet/" + str(mult) + "/" + str(wavelet_absolute_threshold) + "/")
                
                #plot(definition.pad_mask(w_mask, 51)*slice, cmap="hot", norm=colors.Normalize(0, 70), title="Defined dense cores", dpi=300)
                end = time.time()
                print("Running wavelet done")
                print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(end - start)), "\n")
            
            if fourier:
                print("\n", "Running fourier")
                f = self.low_pass_filter_fourier(slice, lp_filter_radius=fourier_lp_filter_radius)
                processed_data.append(slice - f)
                masks.append((slice - f) > fourier_absolute_threshold)
                plot_graphs_and_images_paths.append(catalog_folder_path + "fourier/" + str(mult) + "/" + str(wavelet_absolute_threshold) + "/")
                print("fourier done")
                plot(f, norm=colors.Normalize(0, 160), cmap="hot")
            
            for j in range(len(processed_data)):
                plot_graphs_and_images_path = plot_graphs_and_images_paths[j] + str(X_UPPER*current_slice) + "_" + str(X_UPPER*(current_slice+1) - 1) + "/"
                if plot_threshold:
                    threshold_xs, threshold_ys = self.get_threshold_plot_values(processed_data[j], 0.25, 0.75, 0.25, check_bbox_size, min_dist_between_peaks, length, mult, lowest_peak_height)
                    scatter_plot(threshold_xs, threshold_ys)
                    
                # Create a mask of only the peaks
                print("Creating masks of peaks...")
                start = time.time()
                local_maxes = skimage.feature.peak_local_max(processed_data[j], min_distance=min_dist_between_peaks, labels = definition.pad_mask(masks[j], check_bbox_size))
                peaks_mask = np.zeros(processed_data[j].shape, dtype=bool)
                peaks_mask[local_maxes[:, 0], local_maxes[:, 1]] = True
                end = time.time()
                print("Creating mask of peaks done")
                print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(end - start)), "\n")
                
                if plot_images: padded_peaks_mask = definition.pad_mask(peaks_mask, visual_padding)
                if plot_images: plot(padded_peaks_mask*slice, cmap="hot", norm=colors.Normalize(0, 70), title="Mask", dpi=300)
                
                # Test against definition
                print("Testing against definition...")
                start = time.time()
                dense_cores_mask, def_plot_arr, counts = definition.test_def(processed_data[j], peaks_mask, length, mult, lowest_peak_height, step=2, max_diff=0.005)
                if plot_images: padded_dense_cores_mask = definition.pad_mask(dense_cores_mask, visual_padding)
                dense_cores_and_artefacts_coordinates = list(zip(*np.where(dense_cores_mask == True)[::-1]))
                dense_and_artefacts_y, dense_and_artefacts_x = np.where(dense_cores_mask == True)[::-1]
                dense_and_artefacts_x_compensated = dense_and_artefacts_x + self.limits[2]
                end = time.time()
                print("Testing agains definition done")
                print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(end - start)), "\n")
                
                print("\n", "Dense cores found by definition: ", len(dense_and_artefacts_x),", from ", len(np.where(peaks_mask)[0]))
                
                # Remove artefacts
                print("Removing artefacts...")
                start = time.time()
                lr_min_mask, lr_min_plot_arr, mins_list = artefacts.lr_min(processed_data[j], dense_cores_mask, 50)
                circ_avg_min_mask, circ_avg_min_plot_arr, _ = artefacts.circ_avg_min(processed_data[j], dense_cores_mask, 50)
                artefacts_mask = lr_min_mask | circ_avg_min_mask
                
                dense_cores_mask = dense_cores_mask & np.logical_not(artefacts_mask)
                

   
                end = time.time()
                print("Removing artefacts done")
                print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(end - start)), "\n")

                dense_cores_values = processed_data[j][dense_cores_mask]
                
                dense_cores_coordinates = list(zip(*np.where(dense_cores_mask == True)[::-1]))
                dense_y, dense_x = np.where(dense_cores_mask == True)[::-1]
                print("Dense cores identified:", len(dense_cores_coordinates), "\n")
                # Check artificial cores
                artefacts_coordinates = list(zip(*np.where(artefacts_mask)[::-1]))
                print("Artefacts identified:", len(artefacts_coordinates), "\n")
                
                if insert_artificial_cores:
                    print("Checking found cores versus inserted cores.")
                    num_found = 0
                    for c in self.art_cores_coords[i][:, :2]:
                        
                        if tuple(c)[::-1] in dense_cores_coordinates:
                            num_found += 1
                        else:
                            distances = np.linalg.norm(dense_cores_coordinates-c[::-1], axis=1)
                            min_index = np.argmin(distances)
                            if distances[min_index] < 10:
                                num_found += 1
                    
                    print(f"Found {num_found}/{len(self.art_cores_coords[i])} inserted cores. ({100*num_found/len(self.art_cores_coords[i])}%)")
                    nl = "\n"
                    #print(f"Did not find the following artificial cores:{nl.join(str(c) for c in self.art_cores_coords[i])}")
                
                if insert_artificial_artefacts:
                    print("Checking found artefacts versus inserted artefacts.")
                    num_found_2 = 0
                    for c in self.art_artefacts_coords[i][:, :2]:
                        if tuple(c)[::-1] in artefacts_coordinates:
                            num_found_2 += 1
                        else:
                            #print(artefacts_coordinates-c[::-1])
                            distances = np.linalg.norm(artefacts_coordinates-c[::-1], axis=1)
                            min_index = np.argmin(distances)
                            if distances[min_index] < 10: #51 size of artificial artefact
                                num_found_2 += 1
                    
                    print(f"Found {num_found_2}/{len(self.art_artefacts_coords[i])} inserted artefacts. ({100*num_found_2/len(self.art_artefacts_coords[i])}%)")
                    nl = "\n"
                    #print(f"Did not find the following artificial artefacts:{nl.join(str(c) for c in self.art_artefacts_coords[i])}")
                
                print("Calculating mass and radius...")
                radius = np.array(self.get_radius(slice, dense_x, dense_y), dtype=float)
                mass_list = np.array(self.get_mass(slice, dense_x, dense_y, radius), dtype=float)*0.0081
                scatter_plot(radius*0.02, mass_list, xlabel="radie [pc]", ylabel="massa [M$_{\odot}$]", yscale="log", xscale='log', s=1)
                
                full_artefact_mask = dense_cores_mask & artefacts_mask
                radius_and_artefacts = self.get_radius(slice, dense_and_artefacts_x, dense_and_artefacts_y)
                mass_list_and_artefacts = self.get_mass(slice, dense_and_artefacts_x, dense_and_artefacts_y, radius_and_artefacts)
                print("Calulating mass and radius done")
                
                artefact_flag = []
                for k in dense_cores_and_artefacts_coordinates:
                    if k in artefacts_coordinates:
                       artefact_flag.append(True) 
                    else:
                        artefact_flag.append(False)      
                artefact_flag = np.array(artefact_flag)
                
                if insert_artificial_cores:
                    y = np.array(self.art_cores_coords[0][:, 0], dtype=int)
                    x = np.array(self.art_cores_coords[0][:, 1], dtype=int)
                    art_radius = np.array(self.get_radius(slice, y, x), dtype=float)
                    
                    artificial_mass_list = np.array(self.get_mass(slice, y, x, art_radius), dtype=float)*0.0081 #multiplied to get solar masses
                    scatter_plot(art_radius*0.02, artificial_mass_list, xlabel="radie [pc]", ylabel="massa [M$_{\odot}$]", yscale="log", xscale='log', s=1)
                
                if save:
                    print("Saving")
                    file_name = str(X_UPPER*current_slice) + "_" + str(X_UPPER*(current_slice+1) - 1)
                    #catalog np.array([dense_cores_coordinates, radius, mass_list], dtype=object)
                    #catalog = np.array([list(zip(dense_and_artefacts_x_compensated, dense_and_artefacts_y)), radius_and_artefacts, mass_list_and_artefacts, artefact_flag], dtype=object)
                    radius_and_artefacts = np.array(radius_and_artefacts, dtype=int)
                    catalog = np.array([np.array(range(0, len(dense_and_artefacts_x_compensated)), dtype=np.str), dense_and_artefacts_x_compensated, dense_and_artefacts_y, radius_and_artefacts, mass_list_and_artefacts, artefact_flag], dtype=np.str)
                    catalog = np.hstack([[["Index"], ["X"], ["Y"], ["Radius"], ["Mass"], ["Is artefact"]], np.array(catalog,dtype=np.str)])
                    
                
                    if insert_artificial_cores and insert_artificial_artefacts:
                        performance = f"Artificial cores percentage:{num_found}, arificial artefacts percentage:{num_found_2}"
                        save_file = np.array([parameter_info, performance, catalog], dtype=object)
                    elif insert_artificial_artefacts:
                        performance = f"Arificial artefacts percentage:{num_found_2}"
                        save_file = np.array([parameter_info, performance, catalog], dtype=object)
                    elif insert_artificial_cores:
                        performance = f"Artificial cores percentage:{num_found},"
                        save_file = np.array([parameter_info, performance, catalog], dtype=object)
                    else:
                        save_file = np.array(catalog)
                    Path(plot_graphs_and_images_path).mkdir(parents=True, exist_ok=True)
                    np.save(plot_graphs_and_images_path + file_name, save_file)
                    np.savetxt(plot_graphs_and_images_path + file_name + ".txt", np.transpose(save_file), fmt='%-10s')
                    print("Saved")
                
                # Get data to plot
                if plot_images:
                    padded_dense_cores_mask_no_artefacts = definition.pad_mask(dense_cores_mask, visual_padding)
                    padded_artefacts_mask = definition.pad_mask(artefacts_mask, visual_padding)
                    padded_dense_cores = np.where(padded_dense_cores_mask, slice, slice*0.0)
                    padded_dense_cores_no_artefacts = np.where(padded_dense_cores_mask_no_artefacts, slice, slice*0.0)
                    padded_artefacts = np.where(padded_artefacts_mask, slice, slice*0.0)
                
                #plot_def_and_artefacts(processed_data[j], slice, range(0, 10000), 50, length, mult, lowest_peak_height, def_plot_arr, lr_min_plot_arr, circ_avg_min_plot_arr, onlyArtefacts=True, onlyPos=True)
                
                # Plot images and graphs
                # plot_general((slice, padded_dense_cores, np.where(definition.pad_mask(not_found_mask, visual_padding), slice, 0)), title="Original, Found, Not found")
                if plot_images:
                    plot(padded_dense_cores, cmap="hot", norm=colors.Normalize(0, 70), title="Defined dense cores", dpi=300)
                    plot(padded_dense_cores_no_artefacts, cmap="hot", norm=colors.Normalize(0, 70), title="Defined dense cores - artefacts", dpi=300)
                    plot(padded_artefacts, cmap="hot", norm=colors.Normalize(0, 70), title="Artefacts", dpi=300)
                    #plot_general((slice, padded_dense_cores), title="Original, Found", norm=colors.Normalize(0, 70), dpi=100)
                 
               
                if save_plots_and_images:
                    plot_graphs_and_images(6, 8, 20, processed_data[j], slice, peaks_mask, dense_cores_mask, 50, plot_images=True, title="Classified dense cores", lr_min_artefact=np.array([None]), circ_avg_min_artefact=np.array([None]), avg_graph=np.array([None]), mins_list=mins_list[(circ_avg_min_plot_arr[5] == False) & (lr_min_plot_arr[5] == False)], path=(plot_graphs_and_images_path + "dense_cores/"), plot=False)
                    plot_graphs_and_images(8, 8, 20, processed_data[j], slice, dense_cores_mask | circ_avg_min_mask | lr_min_mask, circ_avg_min_mask | lr_min_mask, 50, plot_images=True, title="Classified artefacts", lr_min_artefact=lr_min_plot_arr[5][(circ_avg_min_plot_arr[5] == True) | (lr_min_plot_arr[5] == True)], circ_avg_min_artefact=circ_avg_min_plot_arr[5][(circ_avg_min_plot_arr[5] == True) | (lr_min_plot_arr[5] == True)], avg_graph=circ_avg_min_plot_arr[4][(circ_avg_min_plot_arr[5] == True) | (lr_min_plot_arr[5] == True)], mins_list=mins_list[(circ_avg_min_plot_arr[5] == True) | (lr_min_plot_arr[5] == True)], path=(plot_graphs_and_images_path + "artefacts/"), plot=False)
                else:
                    plot_graphs_and_images(6, 8, 10, processed_data[j], slice, peaks_mask, dense_cores_mask, 50, plot_images=True, title="Classified dense cores", lr_min_artefact=np.array([None]), circ_avg_min_artefact=np.array([None]), avg_graph=np.array([None]), plot=True, mins_list=counts[def_plot_arr[2]][(circ_avg_min_plot_arr[5] == False) & (lr_min_plot_arr[5] == False)])
                    plot_graphs_and_images(8, 8, 30, processed_data[j], slice, dense_cores_mask | circ_avg_min_mask | lr_min_mask, circ_avg_min_mask | lr_min_mask, 100, plot_images=True, title="Classified artefacts ", lr_min_artefact=lr_min_plot_arr[5][(circ_avg_min_plot_arr[5] == True) | (lr_min_plot_arr[5] == True)], circ_avg_min_artefact=circ_avg_min_plot_arr[5][(circ_avg_min_plot_arr[5] == True) | (lr_min_plot_arr[5] == True)], avg_graph=circ_avg_min_plot_arr[4][(circ_avg_min_plot_arr[5] == True) | (lr_min_plot_arr[5] == True)], mins_list=mins_list[(circ_avg_min_plot_arr[5] == True) | (lr_min_plot_arr[5] == True)], plot=True)
                   
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
            current_slice += 1
            print(current_slice)
        
        
            
if __name__ == "__main__":
    plt.style.use(astropy_mpl_style)
    
    src_path = ""
    catalog_folder_path = ""

    X_LOWER, X_UPPER = 0_000, 12_000
    Y_LOWER, Y_UPPER = 0, 7_000

    sc = Classifier(src_path, [Y_LOWER, Y_UPPER, X_LOWER, X_UPPER], single_slice=True)
    sc.run(True, False, False, insert_artificial_cores=False, insert_artificial_artefacts=False, save=False, compare=False, merge=False, save_plots_and_images=False)

