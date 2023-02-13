import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from matplotlib.colors import LogNorm
from astropy.visualization import astropy_mpl_style

plt.style.use(astropy_mpl_style)
X_LOWER, X_UPPER = 0, 120_000
Y_LOWER, Y_UPPER = 0, 12_000

# Notes:
"""
The data is not square, the top and bottom edges are spikey-ish, making auto-detection of scale tricky
A log colour scale works better than regular
Data can be slices easily as it is a numpy array, using memmap is probably best
The lab computers can load the entire image in about 3 mins
There is only one HDU (the primary one), we could add our masks to the same FITS file
"""


src_path = "source_data/PROMISE-Q1-8micron-filled-v0_3.fits"
with fits.open(src_path, memmap=True) as fits_data:
    fits_data.info()  # Shape is 12,000 x 120,000
    slice_data = fits_data["PRIMARY"].data[ Y_LOWER:Y_UPPER, X_LOWER:X_UPPER]

    plt.figure()
    plt.imshow(slice_data, cmap='twilight', norm=LogNorm())
    plt.colorbar()
    plt.show()