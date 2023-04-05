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
def test_def(data, peaks_mask, length, mult, lowest_val, remove_len=None):
    
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

    stds, means, lengths = get_std_from_matrices(matrices, remove_len)

    filtered_by_def = np.bitwise_and(peak_values - means > stds * mult, peak_values - means > lowest_val)
    peaks_mask[peak_rows, peak_cols] = filtered_by_def
    
    # def_rows, def_cols = np.where(peaks_mask)
    # print("Test on definition done. From " + str(s) + " to " + str(np.sum(peaks_mask)) + " pixels.")
    # mask[rows,cols] = values > stds * mult
    return peaks_mask # peaks_mask

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
    if matrices.shape[2] % 2 != remove_len % 2:
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
        
        for i in range(0, max_remove_size, step):
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


# Plots graphs of given indexes where definition has been tested.
def plot_def(index_arr, x_view, arr, realdata, lr_artefacts_arr, circ_artefacts_arr, onlyPos=False, onlyArtefacts=False, artefact_df_indexing=False):
    data, length, mult, lowest_val, def_rows, def_cols, stds, means, lengths, test = arr
    artefacts_index = 0
    for i in index_arr:
        if i > len(def_rows) - 1:
            break
        
        if (not test[i] and onlyPos):
            if not artefact_df_indexing:
                artefacts_index += 1
            continue
        
        if (onlyArtefacts and not (lr_artefacts_arr[artefacts_index][6] or circ_artefacts_arr[artefacts_index][6])):
            artefacts_index += 1
            continue
        
        
        x_view_left = max(def_cols[i]-x_view, 0)
        x_view_right = min(def_cols[i]+x_view+1, len(arr[0][0]))
        
        xstart = max(def_cols[i]-x_view, 0)
        xend = min(def_cols[i]+x_view+1, len(arr[0][0]))
        
        ystart = max(def_rows[i]-x_view, 0)
        yend = min(def_rows[i]+x_view+1, len(arr[0][0]))
        
        fig = plt.figure(figsize=(20,20), dpi=100)
        gs = fig.add_gridspec(2,3)
        
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(realdata[ystart:yend, xstart:xend], cmap="hot", origin='lower', interpolation="none", norm=Normalize(0, np.max(realdata[ystart:yend, xstart:xend])))
        
        ax = fig.add_subplot(gs[0, 1:])
        
        ax.plot(range(-(x_view_right - x_view_left)//2, (x_view_right - x_view_left)//2), data[def_rows[i], x_view_left:x_view_right], '-')
        
        if not artefact_df_indexing or test[i]:
            ax.plot(range(-(x_view_right - x_view_left)//2, (x_view_right - x_view_left)//2), lr_artefacts_arr[artefacts_index][3], '-')
            left_index = lr_artefacts_arr[artefacts_index][4]
            right_index = lr_artefacts_arr[artefacts_index][5]
            
            if left_index != None and right_index != None:
                left_y = lr_artefacts_arr[artefacts_index][3][left_index]
                left_x = left_index - lr_artefacts_arr[artefacts_index][2] - 1
                
                right_y = lr_artefacts_arr[artefacts_index][3][right_index]
                right_x = right_index - lr_artefacts_arr[artefacts_index][2] - 1
                
                ax.plot(left_x, left_y, 'o', right_x, right_y, 'o')
        
        ax.axhline(y = data[def_rows[i], def_cols[i]], markersize=6, color="black", linewidth=2, linestyle="dashed")
        ax.axhline(y = means[i], color="black")
        #plt.axhline(y = means[i]+lowest_val, color="r")
        ax.axhline(y = means[i]+stds[i]*mult, color="y")
        ax.axhline(y = means[i]+stds[i], color="y", linestyle="dashed")
        ax.axvline(x = -lengths[i]//2, color="black")
        ax.axvline(x = lengths[i]//2, color="black")

        ax = fig.add_subplot(gs[1, 0])
        ax.imshow(data[ystart:yend, xstart:xend], cmap="hot", origin='lower', interpolation="none", norm=Normalize(0, 5))
     
        ax = fig.add_subplot(gs[1, 1])
        ax.plot(range(-(x_view_right - x_view_left)//2, (x_view_right - x_view_left)//2), realdata[def_rows[i], x_view_left:x_view_right])
        ax.axvline(x = -lengths[i]//2, color="black")
        ax.axvline(x = lengths[i]//2, color="black")
     
        ax = fig.add_subplot(gs[1, 2])
        
        if not artefact_df_indexing or test[i]:
            ax.plot(range(0, len(circ_artefacts_arr[artefacts_index][5])), circ_artefacts_arr[artefacts_index][5])
            ax.plot(range(0, len(circ_artefacts_arr[artefacts_index][3])), circ_artefacts_arr[artefacts_index][3])
            ax.plot(circ_artefacts_arr[artefacts_index][4], circ_artefacts_arr[artefacts_index][3][circ_artefacts_arr[artefacts_index][4]], 'o')
            
            fig.suptitle(str(i) + ": " + "(" + str(def_cols[i]) + ", " + str(def_rows[i]) + ") " + str(test[i]) + ", Artefact: " + str(lr_artefacts_arr[artefacts_index][6]) + " " + str(circ_artefacts_arr[artefacts_index][6]), fontsize=20)
        else:
            fig.suptitle(str(i) + ": " + "(" + str(def_cols[i]) + ", " + str(def_rows[i]) + ") " + str(test[i]), fontsize=20)
        
        plt.show()
        
        if not artefact_df_indexing or test[i]:
            artefacts_index += 1

def plot_def_3d(index_arr, x_view, y_view, arr, dpi=100, degree=90):
    data, length, mult, lowest_val, def_rows, def_cols, stds, means, lengths, test = arr
    
    for i in index_arr:
        if i > len(def_rows) - 1:
            break
        
        xstart = max(def_cols[i]-x_view, 0)
        xend = min(def_cols[i]+x_view+1, len(arr[0][0]))
        
        ystart = max(def_rows[i]-y_view, 0)
        yend = min(def_rows[i]+y_view+1, len(arr[0][0]))
        
        data_slice = data[ystart:yend, xstart:xend]
        X = np.arange(xstart, xend, 1)
        Y = np.arange(ystart, yend, 1)
        X, Y = np.meshgrid(X, Y)
        Z = data_slice
        
        fig = plt.figure(figsize=(20,20), dpi=dpi)
        plt.title(str(i) + ": " + "(" + str(def_cols[i]) + ", " + str(def_rows[i]) + ") " + str(test[i]))
        
        for i in range(4):
            ax = fig.add_subplot(2, 2, i+1, projection='3d')
            ax.view_init(elev=30, azim=degree*i)
            # Make data.

            # Plot the surface.
            surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

            # Customize the z axis.
            ax.set_zlim(-1.01, np.max(data_slice)+1)
            ax.zaxis.set_major_locator(LinearLocator(10))
            #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        plt.show()
        
