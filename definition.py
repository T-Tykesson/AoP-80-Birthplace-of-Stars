from skimage.filters import gaussian, threshold_local
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from astropy.io import fits
import numpy as np
from sys import exit

def get_matrices_from_mask(data, mask, length):
    if length % 2 == 0:
        print("Length must be odd!")
        exit()
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

def get_std_from_matrices(matrices):
    return np.std(matrices.reshape(len(matrices), len(matrices[0])*len(matrices[0][0])), axis=1)

def get_mean_from_matrices(matrices):
    return np.mean(matrices.reshape(len(matrices), len(matrices[0])*len(matrices[0][0])), axis=1)

def test_on_def(data, mask, length, mult, lowest_val):
    if (np.sum(mask) == 0):
        print("Mask is blank")
        return mask
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
    stds = get_std_from_matrices(matrices)
    means = get_mean_from_matrices(matrices)
    mask[rows,cols] = np.bitwise_and(means + values > stds * mult, values > lowest_val)
    print("test on definition done")
    #mask[rows,cols] = values > stds * mult
    return mask

def pad_mask(mask, length):
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

def mask_to_data(data, mask):
    rows, cols = np.where(mask == True)
    rows *= len(data)
    rows //= len(mask)
    cols *= len(data[0])
    cols //= len(mask[0])
    new_mask = np.zeros(data.shape, dtype=bool)
    new_mask[rows, cols] = True
    return new_mask
