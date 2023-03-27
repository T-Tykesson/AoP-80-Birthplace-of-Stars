from skimage.filters import gaussian, threshold_local
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, LightSource
from astropy.io import fits
import numpy as np
from matplotlib import cm
from sys import exit

# Returns the indexes of the squares (with a given length) around the 1:s in the mask.
def get_matrices_from_mask(data, mask, length):
    if length % 2 == 0:
        print("Length must be odd!")
        exit()
    s = np.sum(mask)
    rows, cols = np.where(mask)
    rows = np.expand_dims(rows, axis=1)
    cols = np.expand_dims(cols, axis=1)
    row_pad = np.pad(rows, ([0, 0], [length - 1, 0]), 'edge')
    col_pad = np.pad(cols, ([0, 0], [length - 1, 0]), 'edge')
    offsets = np.pad(np.array([range(-(length//2), length//2 + 1)]), ([0, len(row_pad) - 1], [0, 0]), 'edge')
    print("padded")
    row_pad = row_pad + offsets
    row_pad = np.minimum(row_pad, (len(data) - 1) * len(mask) // len(data))
    row_pad = np.maximum(row_pad, 0)
    col_pad = col_pad + offsets
    col_pad = np.minimum(col_pad, (len(data[0]) - 1) * len(mask[0]) // len(data[0]))
    col_pad = np.maximum(col_pad, 0)
    print("Set offsets")
    expanded_rows = np.expand_dims(row_pad, axis=2)
    print("expanded rows")
    expanded_cols = np.pad(np.expand_dims(col_pad, axis=1), ([0,0],[0,length - 1],[0,0]), 'edge')
    print("expanded cols")
    return expanded_rows, expanded_cols

# Returns a mask, based on a given mask, where at every 1, the corresponding pixel 
# in the data satisifies the given definition. Returns also an array
# used to plot where definition has been tested.
def test_on_def(data, mask, length, mult, lowest_val, remove_len=None):
    
    s = np.sum(mask)
    if (s == 0):
        print("Mask is blank")
        return mask
    print("Testing definition on ", s, " pixels.")
    mask = mask.copy()
    rows, cols = np.where(mask == True)
    values = data[rows * len(data) // len(mask), cols * len(data[0]) // len(mask[0])]
    matrices_rows, matrices_cols = get_matrices_from_mask(data, mask, length)
    matrices_rows *= len(data)
    matrices_rows //= len(mask)
    matrices_cols *= len(data[0])
    matrices_cols //= len(mask[0])
    matrices = data[matrices_rows, matrices_cols]
    print("got matrices")
    stds, means, lengths = get_std_from_matrices(matrices, remove_len)
    test = np.bitwise_and(values - means > stds * mult, values - means > lowest_val)
    mask[rows,cols] = test
    
    def_rows, def_cols = np.where(mask)
    print("Test on definition done. From " + str(s) + " to " + str(np.sum(mask)) + " pixels.")
    #mask[rows,cols] = values > stds * mult
    return mask, [data, length, mult, lowest_val, rows, cols, stds, means, lengths, test]

# Returns a mask where there is a square with a given length at every 1 in a given mask.
def pad_mask(mask, length):
    print("Pad mask started.")
    if (np.sum(mask) == 0):
        print("Mask is blank")
        return mask
    mask = mask.copy()
    rows, cols = np.where(mask == True)
    mat_rows, mat_cols = get_matrices_from_mask(mask, mask, length)
    print("got matrices")
    mask[mat_rows, mat_cols] = np.bitwise_or(mask[mat_rows, mat_cols], True)
    print("pad mask done")
    return mask

# Returns sublists of matrices where the center square of a given length is removed. 
def get_sub_lists_from_matrices(matrices, remove_len):
    
    lists = matrices.reshape(len(matrices), len(matrices[0])*len(matrices[0][0]))
    if remove_len == 0: return lists
    mask_matrix = np.pad([[False]], [remove_len - 1, 0], 'edge')
    pad_len = (len(matrices[0]) - remove_len) // 2
    mask_matrix = np.pad(mask_matrix, ([pad_len, pad_len], [pad_len, pad_len]), constant_values=True)
    mask_list = mask_matrix.reshape(len(mask_matrix)*len(mask_matrix[0]))
    #print(len(lists[0]))
    #print(len(mask_list))
    return lists[:, mask_list]

# Returns the converging standard deviations of a list of matrices by removing the center of the matrices,
# or the standard deviations where the center is removed by a given length.
def get_std_from_matrices(matrices, remove_len=None):
    
    max_diff = 0.1
    length = len(matrices[0])
    std_list = np.array([0]*len(matrices))
    mean_list = np.array([0]*len(matrices))
    length_list = np.array([0]*len(matrices))
    
    if remove_len == None:
        
        sub_lists = get_sub_lists_from_matrices(matrices, 0)
        prev_std_from_sub_lists = np.std(sub_lists, axis=1)
        
        for i in range(7, length, 6):
            print(i, " : ", end="")
            sub_lists = get_sub_lists_from_matrices(matrices, i)
            std_from_sub_lists = np.std(sub_lists, axis=1)
            std_diff = prev_std_from_sub_lists - std_from_sub_lists
            
            std_diff_check = np.bitwise_and(np.abs(std_diff) < max_diff, std_list == 0)
            print(std_diff)
            #print(std_diff_check)
            std_list[std_diff_check] = std_from_sub_lists[std_diff_check]
            mean_list[std_diff_check] = np.mean(sub_lists[std_diff_check], axis=1)
            length_list[std_diff_check] = i
            #print(length_list)
            prev_std_from_sub_lists = std_from_sub_lists
            if np.sum(std_list != 0) == len(matrices):
                break
            
    else:
        sub_lists = get_sub_lists_from_matrices(matrices, remove_len)
        std_list = np.std(sub_lists, axis=1)
        mean_list = np.mean(sub_lists, axis=1)
        length_list = np.array([remove_len]*len(matrices))
        #print(length_list)
        
    if (np.sum(std_list == 0) > 0):
        print(str(np.sum(std_list == 0)) + " of " + str(len(std_list)) + " matrices didn't get a converging standard deviation.")
        std_list[std_list == 0] = prev_std_from_sub_lists[std_list == 0]
        mean_list[std_list == 0] = np.mean(sub_lists[std_list == 0], axis=1)
        length_list[std_list == 0] = i
        #print(i)
        
    return std_list, mean_list, length_list


# Plots graphs of given i where definition has been tested.
def plot_def(index_arr, x_view, arr):
    data, length, mult, lowest_val, def_rows, def_cols, stds, means, lengths, test = arr
    for i in index_arr:
        if i > len(def_rows) - 1:
            break
        plt.title(str(i) + ": " + "(" + str(def_cols[i]) + ", " + str(def_rows[i]) + ") " + str(test[i]))
        x_view_left = max(def_cols[i]-x_view, 0)
        x_view_right = min(def_cols[i]+x_view+1, len(arr[0][0]))
        plt.plot(range(-(x_view_right - x_view_left)//2, (x_view_right - x_view_left)//2), data[def_rows[i], x_view_left:x_view_right])
        
        plt.axhline(y = data[def_rows[i], def_cols[i]], markersize=6, color="black", linewidth=2, linestyle="dashed")
        plt.axhline(y = means[i], color="black")
        plt.axhline(y = means[i]+lowest_val, color="r")
        plt.axhline(y = means[i]+stds[i]*mult, color="y")
        plt.axvline(x = -lengths[i]//2, color="black")
        plt.axvline(x = lengths[i]//2, color="black")

        plt.show()
