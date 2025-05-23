"""
Module to manage the training and testing of the ML model, including dataset formatting and performance evaluation.
"""

import pandas as pd
from modules.utils import print_to_log_file

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def get_num_ml_iterations(p, dirs):
    """
    Determine number of ml iterations

    Parameters:
    p: A dictionary containing required parameters.
    """
    
    if p['task'] == 'noise':
        p['total passes'] = len(p['noise fraction'])
        p['number of swept variables'] = 1
        
    elif p['task'] == 'bandwidth':    
        p['total passes'] = len(p['bandwidth'])
        p['number of swept variables'] = 1
        
    elif p['task'] == 'center frequency':    
        p['total passes'] = len(p['center frequency'])
        p['number of swept variables'] = 1
        
    elif p['task'] == 'bandwidth and center frequency':    
        p['total passes'] = len(p['bandwidth']) * len(p['center frequency'])
        p['passes per variable'] = {'bandwidth': len(p['bandwidth']),
                                    'center frequency': len(p['center frequency'])
                                    }
        p['number of swept variables'] = 2
    
    print_to_log_file(dirs['log file'], f"ML iterations requested: {p['total passes']}")
    

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def initialize_new_ml_iteration(p: dict, dirs: dict, iteration_number: int, iteration_inds: list, accuracy_df: pd.DataFrame, f1_df: pd.DataFrame) -> tuple:
    """
    Initializes a new ML iteration by setting up the required parameters.
    
    Parameters:
        p (dict): Configuration parameters.
        iteration_number (int): Current iteration number.
        iteration_inds (list, optional): Indices used for multi-variable iterations.
        accuracy_df (pd.DataFrame): Dataframe for accuracy logging.
        f1_df (pd.DataFrame): Dataframe for F1-score logging.
    
    Returns:
        tuple: (Iteration results dictionary, updated accuracy_df, updated f1_df)
    """
    if p['number of swept variables'] == 1:
        
        print_to_log_file(dirs['log file'],f"|-----------------------------\n|\n|    ITERATION {iteration_number+1} of {p['total passes']}\n|\n|-----------------------------")
        iteration_variable_name = p['scan key']
        iteration_variable_value = p[p['scan key']][iteration_number]
            
        iteration_results = {'scanned item': iteration_variable_name, 
                             'scanned item value': iteration_variable_value
                             }
        # Update summary dataframes
        accuracy_df.at[iteration_number + 1, iteration_variable_name] = iteration_results['scanned item value']
        f1_df.at[iteration_number + 1, iteration_variable_name] = iteration_results['scanned item value']
        
    elif p['number of swept variables'] == 2:
        
        print_to_log_file(dirs['log file'],f"----------------\nIteration {iteration_number+1} of {p['total passes']}")
        
        iteration_variable1_name = p['scan key']['1']
        iteration_variable2_name = p['scan key']['2']
        
        iteration_variable1_value = p[p['scan key']['1']][iteration_inds[0]]
        iteration_variable2_value = p[p['scan key']['2']][iteration_inds[1]]
        
        iteration_results = {'scanned item 1': iteration_variable1_name, 
                             'scanned item 1 value': iteration_variable1_value,
                             'scanned item 2': iteration_variable2_name, 
                             'scanned item 2 value': iteration_variable2_value
                             }
        
        # Update summary dataframes
        accuracy_df.at[iteration_number + 1, p['scan key']['1']] = iteration_results['scanned item 1 value']     
        accuracy_df.at[iteration_number + 1, p['scan key']['2']] = iteration_results['scanned item 2 value']
        f1_df.at[iteration_number + 1, p['scan key']['1']] = iteration_results['scanned item 1 value']
        f1_df.at[iteration_number + 1, p['scan key']['2']] = iteration_results['scanned item 2 value']
        
    
    return iteration_results, accuracy_df, f1_df


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 