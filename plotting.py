
from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from math import ceil
from tqdm import tqdm
from pathlib import Path


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

def scatter_plot(xs, ys, title=None, xlabel=None, ylabel=None, dpi=100, xscale='linear', yscale='linear', s=5):
    fig, ax = plt.subplots(dpi=dpi)
    if title != None: 
        plt.title(f"{title}")
    if xlabel != None:
        ax.set_xlabel(xlabel)
    if ylabel != None:
        ax.set_ylabel(ylabel)
    ax.scatter(xs, ys, s=s)
    ax.set_yscale(xscale)
    ax.set_xscale(yscale)
    plt.xlim(0.01, 1)
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

    
def plot_catalog(path, catalog, data):
    xs = np.array(catalog[1], dtype=int)
    ys = np.array(catalog[2], dtype=int)
    radius_list = catalog[3]
    mass_list = catalog[4]
    artefact_list = np.max(catalog[5:8, :], axis=0)
    unsharp = np.array(catalog[8], dtype=bool)
    wavelet = np.array(catalog[9], dtype=bool)
    fourier = np.array(catalog[10], dtype=bool)
    
    Path(path + "unsharp_mask/dense_cores/").mkdir(parents=True, exist_ok=True)
    Path(path + "unsharp_mask/artefacts/").mkdir(parents=True, exist_ok=True)
    Path(path + "unsharp_mask/artefacts (unsure)/").mkdir(parents=True, exist_ok=True)
    Path(path + "wavelet/dense_cores/").mkdir(parents=True, exist_ok=True)
    Path(path + "wavelet/artefacts/").mkdir(parents=True, exist_ok=True)
    Path(path + "wavelet/artefacts (unsure)/").mkdir(parents=True, exist_ok=True)
    Path(path + "fourier/dense_cores/").mkdir(parents=True, exist_ok=True)
    Path(path + "fourier/artefacts/").mkdir(parents=True, exist_ok=True)
    Path(path + "fourier/artefacts (unsure)/").mkdir(parents=True, exist_ok=True)
    Path(path + "mixed/dense_cores/").mkdir(parents=True, exist_ok=True)
    Path(path + "mixed/artefacts/").mkdir(parents=True, exist_ok=True)
    Path(path + "mixed/artefacts (unsure)/").mkdir(parents=True, exist_ok=True)
    
    view = 50
    for i in range(8290, len(xs)):
        fig = plt.figure(figsize=(15,5), dpi=300)
        gs = fig.add_gridspec(1, 3)
        
        ax = fig.add_subplot(gs[0, 0])
        #print(i)
        #print(ys[i])
        
        xstart = max(xs[i]-view, 0)
        xend = min(xs[i]+view+1, len(data[0]))
        
        ystart = max(ys[i]-view, 0)
        yend = min(ys[i]+view+1, len(data[0]))
        
        ax.imshow(data[ystart:yend,xstart:xend])
        
        
        ax = fig.add_subplot(gs[0, 1])
        ax.plot(range(xstart, xend), data[ys[i], (xstart):(xend)])
        ax.axvline(x=(xs[i] + radius_list[i]//0.02))
        ax.axvline(x=(xs[i] - radius_list[i]//0.02))
        
        ax = fig.add_subplot(gs[0, 2])
        ax.plot(range(ystart, yend), data[ystart:yend, xs[i]])
        ax.axvline(x=(ys[i] + radius_list[i]//0.02))
        ax.axvline(x=(ys[i] - radius_list[i]//0.02))
        
        
        title = ""
        
        if unsharp[i]:
            title += "Unsharp mask "
        if wavelet[i]:
            title += "Wavelet "
        if fourier[i]:
            title += "Fourier"
        
        title += str(mass_list[i]) + " "
        
        if artefact_list[i] == 2:
            title += "Artefact"
        
        if artefact_list[i] == 1:
            title += "Artefact (unsure)"
        
        fig.suptitle(title, fontsize=10)
        
        if unsharp[i] and not wavelet[i] and not fourier[i]:
            if not artefact_list[i]:
                plt.savefig(path + "unsharp_mask/dense_cores/" + str(i))
            elif  artefact_list[i] == 2:
                plt.savefig(path + "unsharp_mask/artefacts/" + str(i))
            elif  artefact_list[i] == 1:
                plt.savefig(path + "unsharp_mask/artefacts (unsure)/" + str(i))
        elif wavelet[i] and not unsharp[i] and not fourier[i]:
            if not artefact_list[i]:
                plt.savefig(path + "wavelet/dense_cores/" + str(i))
            elif artefact_list[i] == 2:
                plt.savefig(path + "wavelet/artefacts/" + str(i))
            elif artefact_list[i] == 1:
                plt.savefig(path + "wavelet/artefacts (unsure)/" + str(i))
        elif fourier[i] and not unsharp[i] and not wavelet[i]:
            if not artefact_list[i]:
                plt.savefig(path + "fourier/dense_cores/" + str(i))
            elif artefact_list[i] == 2:
                plt.savefig(path + "fourier/artefacts/" + str(i))
            elif artefact_list[i] == 1:
                plt.savefig(path + "fourier/artefacts (unsure)/" + str(i))
        else:
            if not artefact_list[i]:
                plt.savefig(path + "mixed/dense_cores/" + str(i))
            elif artefact_list[i] == 2:
                plt.savefig(path + "mixed/artefacts/" + str(i))
            elif artefact_list[i] == 1:
                plt.savefig(path + "mixed/artefacts (unsure)/" + str(i))
        plt.close()
        
        if i % 10 == 0:
            print(i)
        
        
    
    
def plot_graphs_and_images(width, height, amount, data, original_data, mask, dense_core_mask, length, highest_val=None, plot_images=True, title=None, lr_min_artefact=None, circ_avg_min_artefact=None, avg_graph=None, mins_list=None, path=None, plot=True):
    
    index_list = np.array(range(np.sum(mask)))
    index_list = index_list[(dense_core_mask[np.where(mask)]).flatten()]
    
    r, c = np.where(dense_core_mask)
    
    if highest_val != None:
        vals = data[r, c]
        r = r[vals <= highest_val]
        c = c[vals <= highest_val]
        index_list = index_list[vals <= highest_val]
    mult = 1
    if plot_images != (avg_graph != None).any():
        width=width//3
        mult = 3
    else:
        width=width//4
        mult = 4
    if path != None:
        print("plot_graphs_and_images will save plots to path: " + str(path))
    for i in range(amount):
        print("Plotting", i)
        fig = plt.figure(figsize=(20,20), dpi=200)
        gs = fig.add_gridspec(height,width*mult)
        for r_plot in range(height):
            for c_plot in range(width):
                
                ax = fig.add_subplot(gs[r_plot, c_plot*mult])
                
                index = r_plot * width + c_plot + width*height*i
                if index > len(r) - 1:
                    if title != None:
                        fig.suptitle(title + ": " + str(i), fontsize=50)
                    if path != None:
                        print("Saving plot: " + str(i))
                        Path(path).mkdir(parents=True, exist_ok=True)
                        plt.savefig(path + str(i))
                    if plot:
                        plt.show()
                    else:
                        plt.close()
                    return None
                xs = range(c[index]-length//2, c[index]+length//2)
                xs = np.maximum(xs, 0)
                xs = np.minimum(xs, len(data[0]) - 1)
                ys = data[r[index], xs]
                ax.plot(xs, ys)
                ax.axhline(y=data[r[index], c[index]], linestyle="dashed")
                ax.set_xticks([xs[len(xs)//2]-8, xs[len(xs)//2]+8, xs[len(xs)//2]-25, xs[len(xs)//2]+25])
                if (np.array(lr_min_artefact) != None).any():
                    ax.set_title(str(index_list[index]) + " (" + str(c[index]) + " " + str(r[index]) + ") " + str(lr_min_artefact[index]) + " " + str(circ_avg_min_artefact[index]) + str(mins_list[index]), fontsize=12)
                else:
                    ax.set_title(str(index_list[index]) + " (" + str(c[index]) + " " + str(r[index]) + ") " + str(mins_list[index]), fontsize=12)
               
                if plot_images:
                    ax = fig.add_subplot(gs[r_plot, c_plot*mult+mult - 2])
                    ax.imshow(data[(max(r[index]-length//2, 0)):(min(r[index]+length//2, len(data)-1)), (max(c[index]-length//2, 0)):(min(c[index]+length//2, len(data[0])-1))], norm=colors.Normalize(0,1))
                    ax = fig.add_subplot(gs[r_plot, c_plot*mult+mult - 1])
                    ax.imshow(original_data[(max(r[index]-length//2, 0)):(min(r[index]+length//2, len(data)-1)), (max(c[index]-length//2, 0)):(min(c[index]+length//2, len(data[0])-1))], norm=colors.Normalize(0,70))
                    
                if (avg_graph != None).any():
                    ax = fig.add_subplot(gs[r_plot, c_plot*mult+1])
                    ax.plot(range(len(avg_graph[index])), avg_graph[index])
                    ax.set_xticks([0, 9, 15])
                
                    
        if title != None:
            fig.suptitle(title + ", index:" + str(i), fontsize=50)
        
        if path != None:
            print("Saving plot: " + str(i))
            Path(path).mkdir(parents=True, exist_ok=True)
            plt.savefig(path + str(i))
        
        if plot:
            plt.show()
        else:
            plt.close()


def plot_def_and_artefacts(processed_data, original_data, index_arr, x_view, length, mult, lowest_val, def_plot_arr, lr_artefacts_arr, circ_artefacts_arr, sure_artefact_list, onlyPos=True, onlyArtefacts=False, artefact_df_indexing=True):
    
    # Get the lists containing the data to plot
    peak_rows, peak_cols, filtered_by_def, stds, means, lengths = def_plot_arr
    center_xs, center_ys, lr_smoothed_list, first_left_index_list, first_right_index_list, lr_artefact_list = lr_artefacts_arr
    center_xs, center_ys, circ_smoothed_list, circ_smoothed_index_list, avers, circ_artefact_list = circ_artefacts_arr
    
    # Initialize the artefact index
    artefact_index = np.sum(np.array(range(len(filtered_by_def)))[filtered_by_def] <= index_arr[0])
    
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
        
        if (sure_artefact_list[artefact_index][0] == False):
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
            #ax.plot(range((x_view_left - peak_cols[i] + 1), (x_view_right - peak_cols[i] + 1)), lr_smoothed_list[artefact_index][(x_view + x_view_left - peak_cols[i] + 1):(x_view + x_view_right - peak_cols[i] + 1)], '-')
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
            #ax.plot(range(0, len(circ_smoothed_list[artefact_index])), circ_smoothed_list[artefact_index])
            if circ_smoothed_index_list[artefact_index] != None:
                ax.plot(circ_smoothed_index_list[artefact_index], circ_smoothed_list[artefact_index][circ_smoothed_index_list[artefact_index]], 'o')
            
            fig.suptitle(str(i) + ": " + "(" + str(peak_cols[i]) + ", " + str(peak_rows[i]) + ") " + str(filtered_by_def[i]) + ", Artefact: " + str(lr_artefact_list[artefact_index]) + " " + str(circ_artefact_list[artefact_index]) + " " + str(sure_artefact_list[artefact_index]), fontsize=20)
        else:
            fig.suptitle(str(i) + ": " + "(" + str(peak_cols[i]) + ", " + str(peak_rows[i]) + ") " + str(filtered_by_def[i]), fontsize=20)
        
        plt.show()
        
        if not artefact_df_indexing or filtered_by_def[i]:
            artefact_index += 1