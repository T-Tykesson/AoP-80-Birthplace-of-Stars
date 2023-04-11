
from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from math import ceil
from tqdm import tqdm

def plot_general(functions, dpi=60, fig_size=(30,13), title = None, cmap="hot", scale=1, grid=False, colorbar=True, fig_index=None):

    num_cols = 3

    # plt.figure(fig_index, figsize=fig_size, dpi=dpi)
    
    # If more than one functions/imgs are passed, draw them in the same plot
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

def plot(function, dpi=100, fig_size=(30,13), title = None, norm=None, vmin=None, vmax=None, cmap="gray_r", scale=1, grid=False, colorbar=False):
    plt.figure(figsize=fig_size, dpi=dpi)
    ax = plt.gca()
    im = ax.imshow(function, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
    
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
            
def blob_plot_multiple(detection_list, data_list, ylow, yhigh, xlow, xhigh, dpi=100, cmap="inferno"):
    k = 0
    for blob_log_list in tqdm(detection_list):
            title=f"laplacian of gaussian {k}"
            
            fig, ax = plt.subplots(figsize=(90, 30), sharex=True, sharey=True, dpi=dpi)
            ax.set_title(title)
            ax.imshow(data_list[k], cmap=cmap, origin="lower", vmin=(np.mean(data_list[k])-1*np.std(data_list[k])), vmax=(np.mean(data_list[k])+4*np.std(data_list[k])))
            #ax.imshow(final_mask, origin="lower", norm=Normalize(0, 1))
            
            for blob in tqdm(blob_log_list):
                y, x, r = blob
                c = plt.Circle((x, y), 2*r, color="red", linewidth=2, fill=False)
                ax.add_patch(c)
                
            print("Rendering...")
            ax.set_axis_off()
             
            plt.ylim(ylow, yhigh)
            plt.xlim(xlow, xhigh)
            plt.tight_layout()
            plt.show()
            print("Done")
            k += 1
     
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

