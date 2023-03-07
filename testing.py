import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from matplotlib.colors import LogNorm
import numpy as np
from astropy.visualization import astropy_mpl_style

plt.style.use(astropy_mpl_style)
X_LOWER, X_UPPER = 0, None
Y_LOWER, Y_UPPER = 0, None

src_path = "source_data/hedgehog_from_jpg.fits" 

with fits.open(src_path, memmap=True) as fits_data:
    fits_data.info()  # Shape is 12,000 x 120,000
    print(fits_data["PRIMARY"].data.shape)

    # Reshape the image to get [[[R,G,B],[R,G,B]],[[R,G,B],[R,G,B]]] formatted data
    rgb_data = np.stack((fits_data["PRIMARY"].data[0], fits_data["PRIMARY"].data[1], fits_data["PRIMARY"].data[2]), axis=2)
    # Reverse image
    rgb_data = rgb_data[::-1,::-1,:]

    print(rgb_data)
    print(rgb_data.shape)
    plt.figure()
    plt.imshow(rgb_data, cmap='gray')
    plt.colorbar()
    plt.show()