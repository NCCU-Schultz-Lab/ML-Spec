'''
Main code for Bridge-to-experiment-manuscript repo
'''

import os
import pickle
import sys
import time
import numpy as np
import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from modules.data_augmentation import pollute_data
from modules.data_processing import load_central_dataset, classify_central_dataset, format_pytorch_datasets
from modules.debugging_tools import initialize_check_system, clear_iteration_variables
from modules.ml_model import NeuralNet
from modules.results_reporting import save_training_reports, save_iteration_outputs, initialize_training, log_epoch_outcomes, log_testing_outcomes, update_running_dict, update_dataframes_post_training, update_dataframes_post_testing, initialize_running_dict
from modules.training_pipeline import get_num_ml_iterations, initialize_new_ml_iteration
from modules.utils import print_to_log_file, check_inputs, convert_parameter_datatypes, initialize_dataframes, read_input_file, get_git_info
from modules.visualization import save_checked_spectrum

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

if __name__ == "__main__":
    
    verbose = True
    input_path = Path(sys.argv[1]).expanduser().resolve()
    job_name = input_path.stem  # removes .inp
    
    with open(input_path, "r") as f:
        p = read_input_file(f)
    
    dirs = {'working': os.getcwd(),
            'database': f"{p['database path']}",
            'labels': f"{p['labels path']}",
            'outputs': f"{os.path.dirname(input_path)}"}
    
    if not os.path.exists(dirs['database']):
        print(f"ERROR: {dirs['database']} does not exist.")
    
    p['jobname'] = job_name
    dirs['job outputs'] = os.path.join(dirs['outputs'], f"{p['jobname']}")
    dirs['log file'] = os.path.join(dirs['job outputs'], f"{p['jobname']}.mllog")
    os.makedirs(dirs['job outputs'], exist_ok=True)
    
    # set up log file
    with open(dirs['log file'], "w") as f: 
        f.write("Log file initiated.")
    
    print_to_log_file(dirs['log file'], f'''
JOB INFORMATION:
  job name : {p['jobname']}
  working directory: {dirs['working']}
  outputs directory: {dirs['job outputs']}
Git:
  {get_git_info()}
''')
  
    check_inputs(p, dirs)
    convert_parameter_datatypes(p)
    get_num_ml_iterations(p, dirs)
    
    # Load and classify central dataset
    central_data = load_central_dataset(p, dirs) 
    all_classes, class_info_dict = classify_central_dataset(p, dirs, central_data)
    central_data['classes'] = all_classes
    
    # Initialize objects for later
    scan_key, accuracy_df, f1_df = initialize_dataframes(p)
    p['scan key'] = scan_key
    check_spectrum = initialize_check_system(p, central_data)


    torch.manual_seed(p['torch seed'])
    network_details = {'input size': p['input size'], 
                       'hidden size': p['hidden layer size'],
                       'epochs': p['number of epochs'],
                       'batch size': p['batch size'], 
                       'learning rate': p['learning rate'], 
                       'dropout': p['dropout probability']}
    
    # variables to remove at the end of each iteration  
    iteration_variables = ['iteration_data', 
                           'iteration_results',
                           'training_data', 
                           'training_dict', 
                           'train_running_dict',
                           'testing_data', 
                           'testing_dict', 
                           'testing_running_dict',
                           'testing_in_training_dict', 
                           'train_test_running_dict', 
                           'pollute_summary',
                           'train_loader', 
                           'test_loader']
    
# ---------------------- Variable scan --------------------------------

    if p['number of swept variables'] == 1:
    
        for iteration_number in range(p['total passes']):
            
            class_distributions = {'training': [], 'testing': []}
            
            # Initialize the results dictionary for this iteration
            iteration_results, accuracy_df, f1_df = initialize_new_ml_iteration(p = p,
                                                                                dirs = dirs,
                                                                                iteration_number = iteration_number,
                                                                                iteration_inds = None,
                                                                                accuracy_df = accuracy_df,
                                                                                f1_df = f1_df)
            
            # Generate polluted data
            iteration_data, pollute_summary = pollute_data(p = p,
                                                           dirs = dirs,
                                                           central_data = central_data, 
                                                           iteration_inds=[iteration_number])
            
            # Save spectrum if requested
            save_checked_spectrum(p, dirs, check_spectrum, iteration_data, iteration_number)
            
            # Format datasets for Pytorch
            training_data, testing_data, class_distributions = format_pytorch_datasets(p, dirs, class_distributions, iteration_data)
                
            # Set up model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            train_loader = DataLoader(dataset=training_data, batch_size=p['batch size'], shuffle=True)
            test_loader = DataLoader(dataset=testing_data, batch_size=p['batch size'], shuffle=False)
            model = NeuralNet(p['input size'], p['hidden layer size'], class_info_dict['N'], p['dropout probability']).to(device)
            
            flatten = nn.Flatten()  # convert from w1 x w3 (2D) to w1w3 (1D)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=p['learning rate'])
            
            # ----- Train model -----
            
            training_dict, testing_in_training_dict = initialize_training(p, dirs, train_loader, test_loader) 
                
            start_time = time.perf_counter()
            model.train()                                                                               # ensure model is set to training state
            
            for epoch_index in range(p['number of epochs']):
                
                # set up dictionary with running lists (to be updated through the batches)
                training_running_dict = initialize_running_dict(p, dirs, 'training')
    
                # step through batches
                for batch_index, (images, labels, IDs, t2s) in enumerate(train_loader):
                    
                    images = flatten(images).to(device)
                    labels = labels.to(device)        
                    outputs = model(images)             # forward steps through the neurons
                    loss = criterion(outputs, labels)
                    
                    # analyze and record before proceeding
                    update_running_dict(training_running_dict, IDs, t2s, labels, outputs, stage='training', loss=loss)
                    
                    optimizer.zero_grad()               # gradients need to be zeroed out
                    loss.backward()                     # backwards steps to optimize the weights and biases
                    optimizer.step()
                    
                # log outcomes after all batches complete
                log_epoch_outcomes(epoch_index, training_dict, training_running_dict, isTrain=True)
                
                # test once per epoch
                with torch.no_grad():
                    
                    # set up dictionary with running lists (to be updated through the batches)
                    train_test_running_dict = initialize_running_dict(p, dirs, 'testing in training')
                    
                    for batch_index, (images, labels, IDs, t2s) in enumerate(test_loader):
                        
                        images = flatten(images).to(device)
                        labels = labels.to(device)
                        outputs = model(images)
                        update_running_dict(train_test_running_dict, IDs, t2s, labels, outputs, stage='testing in training')
                        
                    # log outcomes after all batches complete
                    log_epoch_outcomes(epoch_index, testing_in_training_dict, train_test_running_dict)
                
                # end epoch
                print_to_log_file(dirs['log file'], f"  completed {epoch_index+1} of {p['number of epochs']} epochs")
                
            # end training
            print_to_log_file(dirs['log file'], f'  Time allocated to training: {np.round((time.perf_counter() - start_time))} sec')
            update_dataframes_post_training(iteration_number, 
                                            accuracy_df, f1_df, 
                                            training_dict, 
                                            testing_in_training_dict)
            
            # save figures for loss and accuracy (if user requested)
            save_training_reports(p, dirs, training_dict, testing_in_training_dict, iteration_number)
            
            # ----- Test model -----
                
            model.eval()            # set the model to evaluation mode
            with torch.no_grad():   # gradients turned off
                
                testing_running_dict = initialize_running_dict(p, dirs, 'testing')
                for batch_index, (images, labels, IDs, t2s) in enumerate(test_loader):
                    
                    images = flatten(images).to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    
                    update_running_dict(testing_running_dict, IDs, t2s, labels, outputs, stage='testing')
                
                # end testing and record results
                testing_dict = log_testing_outcomes(testing_running_dict, class_info_dict, len(test_loader))
                    
            # ----- End iteration: process and save results -----
            update_dataframes_post_testing(p = p, 
                                           dirs = dirs, 
                                           iteration_number = iteration_number, 
                                           iteration_results = iteration_results, 
                                           testing_dict = testing_dict, 
                                           accuracy_df = accuracy_df, 
                                           f1_df = f1_df)
            
            save_iteration_outputs(p = p, 
                                   dirs = dirs,
                                   network_details = network_details, 
                                   classes = class_info_dict, 
                                   class_distributions = class_distributions, 
                                   training_dict = training_dict, 
                                   testing_in_training_dict = testing_in_training_dict, 
                                   testing_dict = testing_dict, 
                                   iteration_number = iteration_number, 
                                   iteration_results = iteration_results, 
                                   pollute_summary = pollute_summary)
            
            for var in iteration_variables:
                if var in globals():
                    globals()[var] = None
                    del globals()[var]

            clear_iteration_variables(iteration_variables)
            gc.collect()
            
    elif p['number of swept variables'] == 2:
        
        count = 0
        for pass_index_bandwidth in range(p['passes per variable']['bandwidth']):
            for pass_index_center_frequency in range(p['passes per variable']['center frequency']):
            
                class_distributions = {'training': [], 'testing': []}
                
                # Initialize the results dictionary for this iteration
                iteration_results, accuracy_df, f1_df = initialize_new_ml_iteration(p = p, 
                                                                                    dirs = dirs,
                                                                                    iteration_number = count,
                                                                                    iteration_inds=[pass_index_bandwidth, pass_index_center_frequency],
                                                                                    accuracy_df = accuracy_df,
                                                                                    f1_df = f1_df)
                
                # Generate polluted data
                iteration_data, pollute_summary = pollute_data(p = p, 
                                                               dirs = dirs,
                                                               central_data = central_data, 
                                                               iteration_inds=[pass_index_bandwidth, pass_index_center_frequency])
                
                # Save spectrum if requested
                save_checked_spectrum(p, dirs, check_spectrum, iteration_data, iteration_number = count)                             
                
                # Format datasets for Pytorch
                training_data, testing_data, class_distributions = format_pytorch_datasets(p, dirs, class_distributions, iteration_data)
                
                # Set up model
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
                train_loader = DataLoader(dataset=training_data, batch_size=p['batch size'], shuffle=True)   # dataloaders automatically shuffle and split data up into batches of requested size
                test_loader = DataLoader(dataset=testing_data, batch_size=p['batch size'], shuffle=False)
                    
                model = NeuralNet(p['input size'], p['hidden layer size'], class_info_dict['N'], p['dropout probability']).to(device)
                
                flatten = nn.Flatten()                                                                      # convert from w1 x w3 (2D) to w1w3 (1D)
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=p['learning rate'])
                
                # ----- Train model -----
                
                training_dict, testing_in_training_dict = initialize_training(p, dirs, train_loader, test_loader) 
                start_time = time.perf_counter()
                model.train()                                                                               # ensure model is set to training state
                
                for epoch_index in range(p['number of epochs']):
                    
                    # set up dictionary with running lists (to be updated through the batches)
                    training_running_dict = initialize_running_dict(p, dirs, 'training')
        
                    # step through batches
                    for batch_index, (images, labels, IDs, t2s) in enumerate(train_loader):
                        
                        images = flatten(images).to(device)
                        labels = labels.to(device)        
                        outputs = model(images)             # forward steps through the neurons
                        loss = criterion(outputs, labels)
                        
                        # analyze and record before proceeding
                        update_running_dict(training_running_dict, IDs, t2s, labels, outputs, stage='training', loss=loss)
                        
                        optimizer.zero_grad()               # gradients need to be zeroed out
                        loss.backward()                     # backwards steps to optimize the weights and biases
                        optimizer.step()
                        
                    # log outcomes after all batches complete
                    log_epoch_outcomes(epoch_index, training_dict, training_running_dict, isTrain=True)
                    
                    # test once per epoch
                    with torch.no_grad():
                        
                        # set up dictionary with running lists (to be updated through the batches)
                        train_test_running_dict = initialize_running_dict(p, dirs, 'testing in training')
                        
                        for batch_index, (images, labels, IDs, t2s) in enumerate(test_loader):
                            
                            images = flatten(images).to(device)
                            labels = labels.to(device)
                            outputs = model(images)
                            update_running_dict(train_test_running_dict, IDs, t2s, labels, outputs, stage='testing in training')
                            
                        # log outcomes after all batches complete
                        log_epoch_outcomes(epoch_index, testing_in_training_dict, train_test_running_dict)
                    
                    # end epoch
                    print_to_log_file(dirs['log file'], f"  completed {epoch_index+1} of {p['number of epochs']} epochs")
                    
                # end training
                print_to_log_file(dirs['log file'], f'  Time allocated to training: {np.round((time.perf_counter() - start_time))} sec')
                update_dataframes_post_training(iteration_number = count, 
                                                accuracy_df = accuracy_df, 
                                                f1_df = f1_df, 
                                                training_dict = training_dict, 
                                                testing_in_training_dict = testing_in_training_dict)
                
                # save figures for loss and accuracy (if user requested)
                save_training_reports(p = p, 
                                      dirs = dirs,
                                      training_dict = training_dict, 
                                      testing_in_training_dict = testing_in_training_dict, 
                                      iteration_number = count)
                
                # ----- Test model -----
                    
                model.eval()            # set the model to evaluation mode
                with torch.no_grad():   # gradients turned off
                    
                    testing_running_dict = initialize_running_dict(p, dirs, 'testing')
                    for batch_index, (images, labels, IDs, t2s) in enumerate(test_loader):
                        
                        images = flatten(images).to(device)
                        labels = labels.to(device)
                        outputs = model(images)
                        
                        update_running_dict(testing_running_dict, IDs, t2s, labels, outputs, stage='testing')
                    
                    # end testing and record results
                    testing_dict = log_testing_outcomes(testing_running_dict, class_info_dict, len(test_loader))
                        
                # ----- End iteration: process and save results -----
                update_dataframes_post_testing(p = p,
                                               dirs = dirs,
                                               iteration_number = count, 
                                               iteration_results = iteration_results, 
                                               testing_dict = testing_dict, 
                                               accuracy_df = accuracy_df, 
                                               f1_df = f1_df)
                
                save_iteration_outputs(p = p, 
                                       dirs = dirs,
                                       network_details = network_details, 
                                       classes = class_info_dict, 
                                       class_distributions = class_distributions, 
                                       training_dict = training_dict, 
                                       testing_in_training_dict = testing_in_training_dict, 
                                       testing_dict = testing_dict, 
                                       iteration_number = count, 
                                       iteration_results = iteration_results, 
                                       pollute_summary = pollute_summary)
                
                for var in iteration_variables:
                    if var in globals():
                        globals()[var] = None
                        del globals()[var]

                clear_iteration_variables(iteration_variables)
                gc.collect()
                count += 1
               
# ---------------------------------------------------------------------------------------

    # Save final results                         
    accuracy_df.to_csv(f"{dirs['job outputs']}/accuracies.csv", index_label = 'Iteration')
    f1_df.to_csv(f"{dirs['job outputs']}/f1scores.csv", index_label = 'Iteration')
    
    # Save inputs
    with open(f"{dirs['job outputs']}/inputs.pkl", 'wb') as f:
        pickle.dump(p, f)
        
    print_to_log_file(dirs['log file'],'''~~~ All iterations complete! ~~~\n\nNow exiting.''')