import get_data
import definition
import artefacts
from us_mask import gaussian_threshold
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import skimage
import skimage.feature
import numpy as np
import wavelet

#%%

def imshow(data, norm_low, norm_high, title=None, cmap="hot", pos=None, indx = None, dpi=100, figsize=(15, 15), yoffset=200):
    plt.figure(figsize=figsize, dpi=dpi)
    if title != None:
        plt.title(title, fontsize=20)
    if pos != None:
        for i in range(len(pos[0])):
            plt.text(pos[1][i], pos[0][i]-yoffset, str(indx[i]), color="w", horizontalalignment='center', fontsize=5)
    plt.imshow(data, cmap=cmap, origin='lower', interpolation="none", norm=Normalize(norm_low, norm_high))
    plt.show()

# Tests definition
def definition_test(data, mask, length, mult, lowest_val, imgs, plots, realdata):
    
    # Create a mask of the peaks
    local_max = skimage.feature.peak_local_max(data, min_distance=5, labels = definition.pad_mask(mask, 11))
    peak_mask = np.zeros(mask.shape, dtype=bool)
    peak_mask[local_max[:, 0], local_max[:, 1]] = True
    
    # Test on definition
    df, arr = definition.test_on_def(data, peak_mask, length, mult, lowest_val)
    
    # Get artefacts
    lr_artefacts_mask, lr_artefacts_arr = artefacts.lr_min(data, df, 25)
    circ_artefacts_mask, circ_artefacts_arr = artefacts.circ_avg_min(data, df, 25)
    artefacts_mask = circ_artefacts_mask | lr_artefacts_mask
    
    if imgs:
        # Calculate pad mask
        pad_df = definition.pad_mask(df, 201)
        pad_df_minus_artefacts = definition.pad_mask(df & ~artefacts_mask, 201)
        pad_artefacts = definition.pad_mask(artefacts_mask, 201)
        
        # Plot
        is_artefact_mask = np.array(np.array(circ_artefacts_arr, dtype=object)[:, 6], dtype=bool)
        imshow(pad_df*realdata, 0, 70, title="Definition", pos=np.where(df), indx = np.array(range(0, len(arr[9])))[arr[9]])
        imshow(pad_df_minus_artefacts*realdata, 0, 70, title="Definition - artefacts", pos=np.where(df & ~artefacts_mask), indx = np.array(range(0, len(arr[9])))[arr[9]][~is_artefact_mask])
        imshow(pad_artefacts*realdata, 0, 70, title="Artefacts", pos=np.where(artefacts_mask), indx = np.array(range(0, len(arr[9])))[arr[9]][is_artefact_mask])
        
    # Plot where definition has been tested
    if plots > 0:
        definition.plot_def(range(0, plots), 25, arr, realdata, lr_artefacts_arr, circ_artefacts_arr, onlyPos=True, onlyArtefacts=False, artefact_df_indexing=True)
    
    tuple_list = []
    
#    for i in range(len(arr[9])): #Skapar lista med koordinater
#        if arr[9][i] == True:
#            tuple_list.append((arr[4][i], arr[5][i]))  

    return df, arr, tuple_list


# Get data
data = get_data.get_data_slice("", 0, 10000, 0, 10000)
g, binary = gaussian_threshold(data, 1, 5)

#w = wavelet.wavelet(data, 3)
#w_sum = w[1][0] + w[1][1] + w[1][2]

# Definition parameters
length = 51
mult = 10
lowest_val = 0

# Resize wavelet and calculate mask
#w_sum = skimage.transform.resize(w_sum, data.shape)
#w_mask = w_sum > 100

# Plot data
imshow(data, 0, 70, title="Original image")
imshow(definition.pad_mask(binary, 201)*data, 0, 70, title="Unsharp mask + threshold")
#imshow(w_sum, 0, 20, dpi=100)

# Test definition
df, arr, tuple_list_1 = definition_test(data - g, binary, length, mult, lowest_val, True, 10, data)

#%%
#artefacts.find_artefacts_from_coords(data, tuple_list_1, 100)

#%%

#df, arr, tuple_list_2 = definition_test(data, w_mask, length, 2, 5, 0)

#artefacts.find_artefacts_from_coords(data, tuple_list_1, 50, 50)


