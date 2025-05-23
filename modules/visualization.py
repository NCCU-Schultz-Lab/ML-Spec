"""
Module to handle all plotting and visualization functions related to spectra and ML results.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from modules.utils import print_to_log_file
import os

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def plot_spectrum(data_dict: dict, save_path: str):
    """
    Generates a contour plot of a 2D spectrum.
    
    Parameters:
        data_dict (dict): Dictionary containing spectral data and metadata. 
            Required keys: 'spectrum', 'w1', 'w3', 'system ID number', 't2'.
        save_path (str): File path to save the generated plot.
    """
    minsize = 16
    medsize = minsize+2
    bigsize = minsize+6
    
    plt.rc('font', size = minsize)          # controls default text sizes
    plt.rc('axes', titlesize = bigsize)     # fontsize of the axes title
    plt.rc('axes', labelsize = medsize)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize = minsize)    # fontsize of the tick labels
    plt.rc('ytick', labelsize = minsize)    # fontsize of the tick labels
    plt.rc('figure', titlesize = bigsize)   # fontsize of the figure title
    ystr = "$\omega_{3} /2 \pi c (10^{3} cm^{-1})$"
    xstr = "$\omega_{1} /2 \pi c (10^{3} cm^{-1})$"
    
    maxval = np.max(np.abs(data_dict['spectrum']))
    
    fig, ax =plt.subplots(ncols=1,nrows=1, figsize=(7, 7))
    ax.contour(data_dict['w1']/1000, data_dict['w3']/1000, data_dict['spectrum'], levels=np.arange(-maxval, maxval, step=0.1*maxval), colors='k', linestyles='dashed', linewidths=0.25)
    ax.contourf(data_dict['w1']/1000, data_dict['w3']/1000, data_dict['spectrum'], levels=np.arange(-maxval, maxval, step=0.01*maxval), cmap='seismic')
    ax.set_ylabel(ystr)
    ax.set_xlabel(xstr)
    ax.set_title(f"{data_dict['system ID number']}, t2 = {data_dict['t2']} fs")
    ax.set_aspect('equal',adjustable='box')
    ax.set_xlim([data_dict['w1'].min()/1000, data_dict['w1'].max()/1000])
    ax.set_ylim([data_dict['w3'].min()/1000, data_dict['w3'].max()/1000])
            
    fig.tight_layout()
    plt.ioff()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    
    pass

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def create_and_save_line_plot(x: np.ndarray, 
                              y: np.ndarray,
                              dirs: dict,
                              save = False,
                              save_dir = None,
                              name = None,
                              title = '', 
                              x_label = 'x', 
                              y_label = 'y',
                              y_limits = None,
                              rot = 0,
                              dpi = 300,
                              width = 8,
                              height = 6,
                              grids = False,
                              x_tick_labels = None):
    """
    Creates and saves a line plot.
    
    Parameters:
        x (np.ndarray): X-axis data.
        y (np.ndarray): Y-axis data.
        title (str): Title of the plot.
        x_label (str): X-axis label.
        y_label (str): Y-axis label.
        ymin (float): Minimum Y-axis value.
        ymax (float): Maximum Y-axis value.
        save_path (str): File path to save the plot.
    """
    
    if save:
        if save_dir is None or name is None:
            print_to_log_file(dirs['log file'], '  ERROR: plot cannot be saved without path and name provided')
        
        else:
    
            plt.figure(figsize=(width, height))
            plt.plot(x, y, linewidth=2)
            plt.title(title, fontsize=20, fontname='DejaVu Sans', fontweight='bold')
            plt.xlabel(x_label, fontsize=18)
            plt.ylabel(y_label, fontsize=18)
            if y_limits:
                plt.ylim(y_limits)
                
            if x_tick_labels is not None:
                plt.xticks(ticks=x, labels=x_tick_labels, rotation=rot, fontsize=14)
            else:
                plt.xticks(fontname='DejaVu Sans', fontsize=16, rotation=rot)
                
            plt.yticks(fontname='DejaVu Sans', fontsize=16)
            plt.grid(grids)
            plt.savefig(f"{save_dir}/{name}.png", dpi=dpi, bbox_inches='tight')
            plt.close()
            
            with open(f"{save_dir}/{name}.pkl", 'wb') as f:
                pickle.dump({'x': x, 'y': y}, f)

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def save_checked_spectrum(p: dict, 
                          dirs: dict, 
                          check_spectrum: dict, 
                          iteration_data: dict, 
                          iteration_number: int,
                          save_data = False):
    """
    Saves a sample 2D spectrum for debugging purposes.
    
    Parameters:
        p (dict): Configuration parameters.
        check_spectrum (dict): Spectrum data to be saved.
        iteration_data (dict): Dataset of the current iteration.
        iteration_number (int): Iteration index.
    """
    if p['save 2D plots'] == True:
        
        save_loc = os.path.join(dirs['job outputs'],"check-spectra")
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)
            
        if (iteration_number+1) % p['spec save interval'] == 0:
            try:
                check_spectrum['spectrum'] = iteration_data['spectra'][check_spectrum['selected system index'],:,:,0]
                plot_spectrum(check_spectrum, 
                              f"{save_loc}/iteration_{iteration_number+1}.png")
                
                with open(f"{save_loc}/iteration_{iteration_number+1}.pkl", 'wb') as f:
                    pickle.dump(check_spectrum, f)
                    
                full_check_spectrum_data = iteration_data['spectra'][check_spectrum['selected system index'],:,:,:]
                with open(f"{save_loc}/iteration_{iteration_number+1}_full.pkl", 'wb') as f:
                    pickle.dump(full_check_spectrum_data, f)
                
            except Exception as e:
                print_to_log_file(dirs['log file'], f'An error occured while generating or saving 2D spectrum: {e}')
    pass

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 