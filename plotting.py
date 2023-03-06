
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_general(function, dpi=60, fig_size=(30,13), title = None, vmin=None, vmax=None, cmap="hot", scale=1, grid=False, colorbar=False):
    mean = np.mean(function)
    std_dev = np.std(function)
    lower = mean - scale*std_dev
    upper = mean + scale*std_dev
    if vmin == None:
        vmin = function.min()
    elif vmin == 1:
        vmin = mean - scale*std_dev
        
    if vmax == None:
        vmax = function.max()
    elif vmax == 1:
        vmax = mean + scale*std_dev
    
    plt.figure(figsize=fig_size, dpi=dpi)
    ax = plt.gca()
    im = ax.imshow(function, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    
    if title != None: 
        plt.title(f"{title}")
    
    if grid:
        plt.grid(alpha=0.05)
    else:
        plt.grid(alpha=0)
        
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
    plt.show()


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
