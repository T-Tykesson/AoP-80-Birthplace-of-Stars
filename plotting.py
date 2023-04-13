
from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from math import ceil
from tqdm import tqdm

def plot_general(functions, dpi=60, fig_size=(30,13), title = None, cmap="hot", scale=1, grid=False, colorbar=True, fig_index=None, norm=None):

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
        
        im = axis[xi][i%num_cols].imshow(func, origin="lower", cmap=cmap, norm=norm)
        
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

def plot_def_and_artefacts(processed_data, original_data, index_arr, x_view, length, mult, lowest_val, def_plot_arr, lr_artefacts_arr, circ_artefacts_arr, onlyPos=True, onlyArtefacts=False, artefact_df_indexing=True):
    
    # Get the lists containing the data to plot
    peak_rows, peak_cols, filtered_by_def, stds, means, lengths = def_plot_arr
    center_xs, center_ys, lr_smoothed_list, first_left_index_list, first_right_index_list, lr_artefact_list = lr_artefacts_arr
    center_xs, center_ys, circ_smoothed_list, circ_smoothed_index_list, avers, circ_artefact_list = circ_artefacts_arr
    
    # Initialize the artefact index
    artefact_index = 0
    
    # Loop through all given indexes
    for i in index_arr:
        
        # Make sure that the index is in range of the data
        if i > len(peak_rows) - 1:
            break
        
        # If the current peak is not defined as a dense core, only dense cores are plotted, and artefacts have not only been checked on dense cores, we update the artefact index.
        if (not filtered_by_def[i] and onlyPos):
            if not artefact_df_indexing:
                artefact_index += 1
            continue
        
        if (onlyArtefacts and not (lr_artefact_list[artefact_index] or circ_artefact_list[artefact_index])):
            artefact_index += 1
            continue
        
        # Get the view lengths
        x_view_left = max(peak_cols[i]-x_view - 1, 0)
        x_view_right = min(peak_cols[i]+x_view, len(processed_data[0]))
        
        xstart = max(peak_cols[i]-x_view, 0)
        xend = min(peak_cols[i]+x_view+1, len(processed_data[0]))
        
        ystart = max(peak_rows[i]-x_view, 0)
        yend = min(peak_rows[i]+x_view+1, len(processed_data[0]))
        
        # Get figure and gridspec
        fig = plt.figure(figsize=(20,20), dpi=100)
        gs = fig.add_gridspec(2,3)
        
        # Plot original data around peak
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(original_data[ystart:yend, xstart:xend], cmap="hot", origin='lower', interpolation="none", norm=colors.Normalize(0, np.max(original_data[ystart:yend, xstart:xend])))
        
        # Plot a horizontal slice of the processed data around the peak 
        ax = fig.add_subplot(gs[0, 1:])
        ax.plot(range((x_view_left - peak_cols[i]), (x_view_right - peak_cols[i])), processed_data[peak_rows[i], x_view_left:x_view_right], '-')
        
        # Plot the smoothed data for detecting artefacts, and its first right and left min points.
        if not artefact_df_indexing or filtered_by_def[i]:
            ax.plot(range((x_view_left - peak_cols[i] + 1), (x_view_right - peak_cols[i] + 1)), lr_smoothed_list[artefact_index][(x_view + x_view_left - peak_cols[i] + 1):(x_view + x_view_right - peak_cols[i] + 1)], '-')
            left_index = first_left_index_list[artefact_index]
            right_index = first_right_index_list[artefact_index]
            
            if left_index != None and right_index != None:
                left_y = lr_smoothed_list[artefact_index][left_index]
                left_x = left_index + (x_view_left - peak_cols[i] + 1)
                
                right_y = lr_smoothed_list[artefact_index][right_index]
                right_x = right_index + (x_view_left - peak_cols[i] + 1)
                
                ax.plot(left_x, left_y, 'o', right_x, right_y, 'o')
                
        # Plot lines how definition is tested 
        ax.axhline(y = processed_data[peak_rows[i], peak_cols[i]], markersize=6, color="black", linewidth=2, linestyle="dashed")
        ax.axhline(y = means[i], color="black")
        if lowest_val != 0:
            plt.axhline(y = means[i]+lowest_val, color="r")
        ax.axhline(y = means[i]+stds[i]*mult, color="y")
        ax.axhline(y = means[i]+stds[i], color="y", linestyle="dashed")
        ax.axvline(x = -lengths[i]//2, color="black")
        ax.axvline(x = lengths[i]//2, color="black")
        
        # Plot the processed data around the peak
        ax = fig.add_subplot(gs[1, 0])
        ax.imshow(processed_data[ystart:yend, xstart:xend], cmap="hot", origin='lower', interpolation="none", norm=colors.Normalize(0, 5))
        
        # Plot a horizontal slice of the original data around the peak
        ax = fig.add_subplot(gs[1, 1])
        ax.plot(range(-(x_view_right - x_view_left)//2, (x_view_right - x_view_left)//2), original_data[peak_rows[i], x_view_left:x_view_right])
        ax.axvline(x = -lengths[i]//2, color="black")
        ax.axvline(x = lengths[i]//2, color="black")
        
        # Plot the circle average + smoothed graph
        ax = fig.add_subplot(gs[1, 2])
        
        if not artefact_df_indexing or filtered_by_def[i]:
            ax.plot(range(0, len(avers[artefact_index])), avers[artefact_index])
            ax.plot(range(0, len(circ_smoothed_list[artefact_index])), circ_smoothed_list[artefact_index])
            if circ_smoothed_index_list[artefact_index] != None:
                ax.plot(circ_smoothed_index_list[artefact_index], circ_smoothed_list[artefact_index][circ_smoothed_index_list[artefact_index]], 'o')
            
            fig.suptitle(str(i) + ": " + "(" + str(peak_cols[i]) + ", " + str(peak_rows[i]) + ") " + str(filtered_by_def[i]) + ", Artefact: " + str(lr_artefact_list[artefact_index]) + " " + str(circ_artefact_list[artefact_index]), fontsize=20)
        else:
            fig.suptitle(str(i) + ": " + "(" + str(peak_cols[i]) + ", " + str(peak_rows[i]) + ") " + str(filtered_by_def[i]), fontsize=20)
        
        plt.show()
        
        if not artefact_df_indexing or filtered_by_def[i]:
            artefact_index += 1