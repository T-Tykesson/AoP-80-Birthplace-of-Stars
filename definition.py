from skimage.filters import gaussian, threshold_local
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, LightSource
from astropy.io import fits
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from sys import exit
import plotting


# Returns the indexes of the squares (with a given size) around the 1:s in the mask.
def create_box_around_peaks(peaks_mask, size):
    if size % 2 == 0:
        raise ValueError("Size must be odd!")
        
    rows, cols = np.where(peaks_mask == True)
    rows = np.expand_dims(rows, axis=1)
    cols = np.expand_dims(cols, axis=1)

    row_pad = np.pad(rows, ([0, 0], [size - 1, 0]), 'edge')
    col_pad = np.pad(cols, ([0, 0], [size - 1, 0]), 'edge')
    offsets = np.pad(np.array([range(-(size//2), size//2 + 1)]), ([0, len(row_pad) - 1], [0, 0]), 'edge')

    row_pad = row_pad + offsets
    row_pad = np.minimum(row_pad, (len(peaks_mask) - 1))
    row_pad = np.maximum(row_pad, 0)

    col_pad = col_pad + offsets
    col_pad = np.minimum(col_pad, (len(peaks_mask[0]) - 1))
    col_pad = np.maximum(col_pad, 0)

    expanded_rows = np.expand_dims(row_pad, axis=2)
    expanded_cols = np.pad(np.expand_dims(col_pad, axis=1), ([0,0],[0,size - 1],[0,0]), 'edge')

    return expanded_rows, expanded_cols

# Returns a mask, based on a given mask of peaks, where at every 1, the corresponding pixel 
# in the data satisifies the given definition.
def test_def(data, peaks_mask, length, mult, lowest_val, remove_len=None, max_diff=0.1, step=6):
    
    s = np.sum(peaks_mask)
    if (s == 0):
        print("Mask is blank")
        return peaks_mask
    print("Testing definition on ", s, " pixels.")
    peaks_mask = peaks_mask.copy()
    peak_rows, peak_cols = np.where(peaks_mask == True)
    peak_values = data[peak_rows, peak_cols]  # Whats the point of this?
    
    expanded_matrices_rows, expanded_matrices_cols = create_box_around_peaks(peaks_mask, length)
    
    # Because data and peaks_mask will have the same shape, the following rows do nothing
    # matrices_rows *= len(data)
    # matrices_rows //= len(peaks_mask)
    # matrices_cols *= len(data[0])
    # matrices_cols //= len(peaks_mask[0])

    matrices = data[expanded_matrices_rows, expanded_matrices_cols]

    stds, means, lengths = get_std_from_matrices(matrices, remove_len, max_diff=max_diff, step=step)

    filtered_by_def = np.bitwise_and(peak_values - means > stds * mult, peak_values - means > lowest_val)
    peaks_mask[peak_rows, peak_cols] = filtered_by_def
    
    # def_rows, def_cols = np.where(peaks_mask)
    # print("Test on definition done. From " + str(s) + " to " + str(np.sum(peaks_mask)) + " pixels.")
    # mask[rows,cols] = values > stds * mult
    return peaks_mask, [peak_rows, peak_cols, filtered_by_def, stds, means, lengths] # peaks_mask

# Returns a mask where there is a square with a given length at every 1 in a given mask.
def pad_mask(mask, length):
    if (np.sum(mask) == 0):
        print("Mask is blank")
        return mask
    mask = mask.copy()
    
    mat_rows, mat_cols = create_box_around_peaks(mask, length)
    mask[mat_rows, mat_cols] = True
    return mask

# Returns sublists of flattened matrices where the center square of a given size (len x len) is removed. 
def remove_centre_from_matrices_and_flatten(matrices: np.ndarray, remove_len: int):
    if matrices.shape[2] % 2 != remove_len % 2 and remove_len != 0:
        raise ValueError("Matrix width/height and remove_len must have matching parity.")
    
    flattened_values = matrices.reshape(len(matrices), len(matrices[0])*len(matrices[0][0]))

    if remove_len == 0: return flattened_values

    mask_matrix = np.zeros((remove_len, remove_len), dtype=bool)
    mask_matrix = np.pad(mask_matrix, pad_width=((matrices[0].shape[0] - remove_len) // 2), mode="constant", constant_values=True)
    
    mask_flattened = mask_matrix.reshape(len(mask_matrix)*len(mask_matrix[0]))
    return flattened_values[:, mask_flattened]

# Returns the converging standard deviations of a list of matrices by removing an expanding portion from the center of the matrices until a defined difference limit is reached,
# or the standard deviations where the center is removed by a given size.
def get_std_from_matrices(matrices, remove_size=None, max_diff=0.1, step=6 ):
    
    max_remove_size = len(matrices[0])  # TODO: This can potenitally remove the entire box
    std_list = np.array([0.0]*len(matrices))
    mean_list = np.array([0.0]*len(matrices))
    length_list = np.array([0.0]*len(matrices))
    
    if remove_size == None:
        
        sub_lists = remove_centre_from_matrices_and_flatten(matrices, remove_len=0)
        prev_std_from_sub_lists = np.std(sub_lists, axis=1)
        
        for i in range(step + 1, max_remove_size, step):
            sub_lists = remove_centre_from_matrices_and_flatten(matrices, remove_len=i)
            std_from_sub_lists = np.std(sub_lists, axis=1)
            std_diff = prev_std_from_sub_lists - std_from_sub_lists
            
            std_diff_check = np.bitwise_and(np.abs(std_diff) < max_diff, std_list == 0)
            
            prev_std_from_sub_lists = std_from_sub_lists

            std_list[std_diff_check] = std_from_sub_lists[std_diff_check]
            mean_list[std_diff_check] = np.mean(sub_lists[std_diff_check], axis=1)
            length_list[std_diff_check] = i

            if np.sum(std_list != 0) == len(std_list):
                break
            
    else:
        sub_lists = remove_centre_from_matrices_and_flatten(matrices, remove_size)
        std_list = np.std(sub_lists, axis=1)
        mean_list = np.mean(sub_lists, axis=1)
        length_list = np.array([remove_size]*len(matrices))
        
    if (np.sum(std_list == 0) > 0):
        print(str(np.sum(std_list == 0)) + " of " + str(len(std_list)) + " matrices didn't get a converging standard deviation.")
        mean_list[std_list == 0] = np.mean(sub_lists[std_list == 0], axis=1)
        length_list[std_list == 0] = i
        std_list[std_list == 0] = prev_std_from_sub_lists[std_list == 0]
        print(std_list == 0)
        
    return std_list, mean_list, length_list

