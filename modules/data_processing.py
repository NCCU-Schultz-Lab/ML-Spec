"""
Module to handle dataset loading, transformations, and classification.
"""

import numpy as np
import pandas as pd
import pickle
from modules.ml_model import PyTorchDataset
from modules.utils import print_to_log_file
from collections import Counter

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def classify_central_dataset(p: dict, dirs: dict, central_data: dict) -> tuple:
    """
    Classify datasets into predefined electronic coupling classes.

    Parameters:
        p (dict): Configuration parameters containing class boundaries.
        central_data (dict): Loaded central dataset.

    Returns:
        tuple: (Updated central dataset, classification metadata)
    """
    
    numClasses = len(p['class bounds']) - 1
    class_info_dict = {'N': numClasses, 
                         'bounds': [(p['class bounds'][i], p['class bounds'][i+1]) for i in range(numClasses)],
                         'numbers': np.linspace(0, numClasses-1, numClasses)
                         }
    
    labelsDf = pd.read_csv(dirs['labels'], index_col=0).loc[central_data['system ID numbers'], :]
    all_classes = []
    for system in labelsDf.index:
        J = labelsDf.loc[system, "J_Coul"]
        for i, (lower, upper) in enumerate(class_info_dict['bounds']):
            if lower <= J < upper:
                all_classes.append(i)
                break
    
    central_data['classes'] = np.array(all_classes)
    
    return np.array(all_classes), class_info_dict

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def format_pytorch_datasets(p: dict, dirs: dict, class_distributions: dict, iteration_data: dict) -> tuple:
    """
    Prepares datasets for PyTorch training and testing.
    
    Parameters:
        p (dict): Configuration parameters.
        dirs (dict): Dictionary containing directory paths.
        class_distributions (dict): Dictionary storing the distribution of classes in training/testing.
        iteration_data (dict): The dataset for the current iteration.
    
    Returns:
        tuple: (Training dataset, Testing dataset, Updated class distributions)
    """
    
    print_to_log_file(dirs['log file'],'Formatting data for Pytorch...')
    
    training_data = PyTorchDataset(iteration_data, p, isTrain=True)
    testing_data = PyTorchDataset(iteration_data, p, isTrain=False)
    
    total_number_spectra = len(testing_data) + len(training_data)
    print_to_log_file(dirs['log file'], f'  total number of spectra for training and testing = {total_number_spectra}')
    
    # Log class distribution
    training_classes = []
    for i in range(len(training_data)):
       check_imgs, check_labels, check_IDs, check_t2s = training_data[i]
       training_classes.append(check_labels.item())

    testing_classes = []
    for i in range(len(testing_data)):
       check_imgs, check_labels, check_IDs, check_t2s = testing_data[i]
       testing_classes.append(check_labels.item())
    
    class_distributions['training'] = Counter(training_classes)
    class_distributions['testing'] = Counter(testing_classes)
    
    return training_data, testing_data, class_distributions

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def load_central_dataset(p: dict, dirs: dict) -> dict:
    """
    Load and preprocess the central dataset for machine learning.

    Parameters:
        p (dict): Configuration parameters.
        dirs (str): Dictionary of relevant paths.

    Returns:
        dict: Central dataset with system indicies, spectra, and frequency axes.
    """
    
    import os
    import time
    import math
    
    method = p['data loading method']
    
        # method (str):
        # filename (str): For single file loading, this is the name of the .pkl containing the full dataset.
        #                 For iterative loading, this is the name of the .txt containing the run ID list
    
    print_to_log_file(dirs['log file'], " ~~~~ LOADING DATA ~~~~ ")
    start_time = time.perf_counter()
    
    if method == "single file":
        
        filename = p['data file name']
    
        with open(os.path.join(dirs['database'], f'{filename}.pkl'), "rb") as pklfile: 
            dataset = pickle.load(pklfile)
            
        central_data = {'spectra': np.array(dataset['system spectra']), 
                        'w1': np.array(dataset['system w1 axis']), 
                        'w3': np.array(dataset['system w3 axis']),
                        't2': np.array(dataset['system t2 value']),
                        'number of systems': len(dataset['system w1 axis']),
                        'nt2': len(dataset['system t2 value'][0]),
                        'nw1': len(dataset['system w1 axis'][0]),
                        'nw3': len(dataset['system w3 axis'][0]),
                        'system ID numbers': np.array(dataset['system index'])
                        }
        
    elif method == "iterative":
        
        labelsDf = pd.read_csv(dirs['labels'], index_col=0)
        central_ID_numbers = labelsDf.index
        
        central_data = {'system ID numbers': np.array(central_ID_numbers), 
                        'custom system IDs': [f"{str(i)}" for i in central_ID_numbers],
                        'number of systems': len(central_ID_numbers)}
        
        data_full = list()   # final shape = [Number of systems, w1, w3, t2]
        w1 = list()
        w3 = list()
        t2 = list()
        message_interval = math.ceil(central_data['number of systems'] // 10)
        for index, system_ID in enumerate(central_data['custom system IDs']):
            data_temp, w1_temp, w3_temp, t2_temp = load_system_data(dirs['database'], system_ID)
            
            if p['t2 truncate'] == True:
                data_temp = data_temp[:,:,0:int(p['t2 truncate index'])]
                t2_temp = t2_temp[0:int(p['t2 truncate index'])]
            
            data_full.append(data_temp)
            w1.append(w1_temp)
            w3.append(w3_temp)
            t2.append(t2_temp)
            if (index+1) % message_interval == 0:
                completion_percentage = 100*(index+1)/central_data['number of systems']
                print_to_log_file(dirs['log file'], f"{completion_percentage:.0f}% loaded")
        
        central_data.update({'spectra': np.array(data_full), 
                             'w1': np.array(w1), 
                             'w3': np.array(w3),
                             't2': np.array(t2), 
                             'nw1': len(w1[0].squeeze()), 
                             'nw3': len(w3[0].squeeze()), 
                             'nt2': len(t2[0].squeeze())}
                            )
        
    else:
        raise Exception("ERROR: Invalid data loading protocol requested. Options: 'single file', 'iterative'")
        
    time_elapsed = np.round((time.perf_counter() - start_time))
    print_to_log_file(dirs['log file'], f'''Finished loading data after {time_elapsed} sec:
    Number of systems: {central_data['number of systems']} 
    Data shape: {central_data['spectra'].shape}
    loading time per system: {np.round(1000*time_elapsed/central_data['number of systems'], decimals = 2)} ms per spectrum\n''')
    
    return central_data

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def load_system_data(database_path, sys_ID):
    '''
    DESCRIPTION: loads system with specific run name from the directory data_path
    INPUTS: 
        database_path ... path to data
        sys_ID .......... ID number for system
        
    RETURNS:
        idata............ array of shape [1, nw1, nw3, nt2] containing 2DES spectra
    '''
    
    import os
    
    filepath = os.path.join(database_path, f'{sys_ID}.pkl')
    with open(filepath, "rb") as pklfile:
        fullEntry = pickle.load(pklfile)

    idata = fullEntry["Rw1w3Absorptive"]
    w1 = fullEntry['w1']
    w3 = fullEntry['w3']
    t2 = fullEntry['t2']
    
    return idata, w1, w3, t2