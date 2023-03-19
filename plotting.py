
from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from math import ceil

def plot_general(functions, dpi=60, fig_size=(30,13), title = None, cmap="hot", scale=1, grid=False, colorbar=True, fig_index=None):

    num_cols = 3

    # plt.figure(fig_index, figsize=fig_size, dpi=dpi)
    
    # If two functions/imgs are passed, draw them both in the same plot
    if not isinstance(functions, tuple):
        functions = (functions,)
    
    fig, axis = plt.subplots(ceil(len(functions)/3), min(len(functions), 3), figsize=fig_size, dpi=dpi, squeeze=False)
    
    if title != None: 
        fig.suptitle(f"{title}")
    
    xi = -1
    for i, func in enumerate(functions):
        if (i%num_cols == 0):
            xi += 1
        
        im = axis[xi][i%num_cols].imshow(func, origin="lower", cmap=cmap)
        
        axis[xi][i%num_cols].grid(alpha=0.05) if grid else axis[xi][i%num_cols].grid(alpha=0)

        if colorbar:
            divider = make_axes_locatable(axis[xi][i%num_cols])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
    plt.draw()


def plot_figure(func, title, norm=None, dpi=None):
    plt.figure()
    if dpi:
        plt.rcParams['figure.dpi'] = dpi
        
    
    plt.title(f"{title}")
    plt.grid(alpha=0.05)
    ax = plt.gca()
    #pcm = ax.pcolor(func, norm=colors.LogNorm(vmin=func.min(), vmax=func.max()))
    im = ax.imshow(func, norm=norm, cmap='gnuplot')
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
       
    plt.colorbar(im, cax=cax)
    plt.show()


def plot_image(func, title, save=False, dpi=None):
    plt.figure()
    plt.axis('off')
    
    if dpi:
        plt.rcParams['figure.dpi'] = dpi
    ax = plt.gca()
    im = ax.imshow(func, norm=colors.PowerNorm(gamma=0.5), cmap='gnuplot')
    
    
    if save:
        plt.savefig(f"{title}.png", dpi=300, bbox_inches=0)
    plt.show()


def plot_spectrum(im_fft):
    # A logarithmic colormap
    plt.imshow(np.abs(im_fft), norm=colors.LogNorm(vmin=5))
    plt.colorbar()