import plotting
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

full_data = False #Ändra till false om man inte vill dela upp hela datan
file_path = "Q1-latest-whigal-85.fits"

if full_data:
    data_list = []
    xslice = 10000
    yslice = 7000

    with fits.open(file_path, use_fsspec=True, fsspec_kwargs={"anon": True}) as hdul:  
       #cutout = hdul[0].section[0:1750, 0:2500] 
        for i in tqdm(range(0, int(7000/yslice))):
            for j in tqdm(range(0, int(120000/xslice))):
                cutout = hdul[0].section[yslice*i:yslice*i + yslice, xslice*j:xslice*j+xslice]
                data_list.append(cutout)
else:
    data = fits.getdata(file_path)
    x_low = 11840*5
    x_high = 11875*5
    y_low = 725*5
    y_high = 755*5
    data = data[y_low:y_high, x_low:x_high]
    

maxim = data.argmax()
index = np.unravel_index(maxim, data.shape)
print(index)
plotting.plot_figure(data,"Artefakt")

size = 40

datap = data[index[0], (index[1]-size):(index[1]+size)]
X = np.linspace(0,2*size-1, num=2*size)

peaksh, _ = find_peaks(-datap)
peaksl, _ = find_peaks(datap)


print(size-peaksl)
print(size-peaksh)

plt.plot(peaksh, datap[peaksh], "x")
plt.plot(peaksl, datap[peaksl], 'x')
plt.plot(X,datap)
plt.show()


'''
    x_low = 15470*5
    x_high = 15495*5
    y_low = 575*5
    y_high = 595*5

    x_low = 11540*5
    x_high = 11565*5
    y_low = 636*5
    y_high = 656*5
    
    x_low = 11840*5
    x_high = 11875*5
    y_low = 725*5
    y_high = 755*5
'''
