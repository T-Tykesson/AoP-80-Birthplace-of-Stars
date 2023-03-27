import get_data
import definition
from us_mask import gaussian_threshold
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import skimage
import numpy as np
import wavelet

def imshow(data, norm_low, norm_high, cmap="hot", dpi=15, figsize=(15, 15)):
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(data, cmap=cmap, origin='lower', interpolation="none", norm=Normalize(norm_low, norm_high))
    plt.show()

# Tests definition
def definition_test(data, mask, length, mult, lowest_val, plots):
    
    # Create a mask of the peaks
    local_max = skimage.feature.peak_local_max(data, min_distance=10, labels = definition.pad_mask(mask, 11))
    peak_mask = np.zeros(mask.shape, dtype=bool)
    peak_mask[local_max[:, 0], local_max[:, 1]] = True
    
    # Test on definition
    df, arr = definition.test_on_def(data, peak_mask, length, mult, lowest_val)
    
    # Plot where definition has been tested
    if plots > 0:
        definition.plot_def(range(0, plots), 100, arr)
    
    # Calculate pad mask
    pad_mask = definition.pad_mask(peak_mask, 201)
    pad_df = definition.pad_mask(df, 201)
    
    # Plot 
    imshow(pad_mask*data + g*~pad_mask*0.3, 0, 70, dpi=100)
    imshow(pad_df*data + g*~pad_df*0.3, 0, 70, dpi=100)
    imshow(pad_df*data + g*~pad_df*0.3, 0, 200, dpi=100)

# Get data
data = get_data.get_data_slice("", 0, 10000, 0, 10000)
g, binary = gaussian_threshold(data, 1, 5)
w = wavelet.wavelet(data, 3)
w_sum = w[1][0] + w[1][1] + w[1][2]

# Definition parameters
length = 201
mult = 3
lowest_val = 100

# Resize wavelet and calculate mask
w_sum = skimage.transform.resize(w_sum, data.shape)
w_mask = w_sum > 100

# Plot data
imshow(data, 0, 70, dpi=100)

# Test definition
definition_test(data, binary, length, mult, lowest_val, 0)
definition_test(data, w_mask, length, mult, lowest_val, 0)


