# %% Import Libraries

import os
import sys
import warnings
import argparse
import json
from Amazonia_Legal_RO import AMAZON_RO
from Amazonia_Legal_PA import AMAZON_PA
from Cerrado_Biome_MA import CERRADO_MA
import SharedParameters
import Main_Train_FC114, Main_Test_FC114
import Main_Compute_Metrics_05, Main_Compute_Average_Metrics_MT
import tensorflow as tf

# %% Limit GPU Memory or, in case of no gpu available, limit number of threads used

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')

memory_limit = 20480
num_threads = 1

if gpus:
    print(f"Limiting memory on first GPU to {memory_limit} MB")
    tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)])

print("Limiting number of threads to: ", num_threads) 
tf.config.threading.set_inter_op_parallelism_threads(num_threads)
tf.config.threading.set_intra_op_parallelism_threads(num_threads)

    
# %% Create parser with function

# An args list can be passed optionally
def parse_args(args=None):    
    parser = argparse.ArgumentParser(description='')
    
    # Train or not
    parser.add_argument('--train', dest='train', type=eval, choices=[True, False], default=True, help='whether to run train script or not')
    
    # Test or not
    parser.add_argument('--test', dest='test', type=eval, choices=[True, False], default=True, help='whether to run test script or not')

    # Evaluate metrics or not
    parser.add_argument('--metrics', dest='metrics', type=eval, choices=[True, False], default=True, help='whether to run metrics script or not')

    # Evaluate metrics average or not
    parser.add_argument('--metrics_avg', dest='metrics_avg', type=eval, choices=[True, False], default=True, help='whether to run metrics avg script or not')

    # Running argument
    parser.add_argument('--running_in', dest='running_in', type=str, default='Datarmor_Interactive', help='Decide wether the script will be running')
    
    # parser.add_argument('--phase', dest = 'phase', type = str, default = 'train', help = 'Decide wether the phase: Train|Test will be running')
    
    return parser.parse_args(args=args)

# %% Main function to execute

def main(argv=None):
    # Command line argv is passed if not specified
    if argv is None:
        argv = sys.argv
        
    # Empty list is passed if argv is [''] (interactive prompt) or
    # list with unique element (the script alone is executed).
    # So the defaults are assumed
    args = parse_args(args=argv[1:]) 
    
    # Print args passed
    print("Module name: ", __name__)
    print("Args: ")
    print(json.dumps(vars(args), indent=4), end='\n\n')
    
    # Decide what to do with running_in parameter
    if args.running_in == 'Datarmor_Interactive':
        Train_MAIN_COMMAND = SharedParameters.Train_MAIN_COMMAND
        Test_MAIN_COMMAND = SharedParameters.Test_MAIN_COMMAND
        Metrics_05_MAIN_COMMAND = SharedParameters.Metrics_05_MAIN_COMMAND
        Metrics_th_MAIN_COMMAND = SharedParameters.Metrics_th_MAIN_COMMAND
    if args.running_in == 'Datarmor_PBS':
        Train_MAIN_COMMAND = SharedParameters.Full_Path_Train_MAIN_COMMAND
        Test_MAIN_COMMAND = SharedParameters.Full_Path_Test_MAIN_COMMAND
        Metrics_05_MAIN_COMMAND = SharedParameters.Full_Path_Metrics_05_MAIN_COMMAND
        Metrics_th_MAIN_COMMAND = SharedParameters.Full_Path_Metrics_th_MAIN_COMMAND
    
    # Ignore warnings
    warnings.filterwarnings("ignore")
    
    # Main results path
    Checkpoint_Results_MAIN_PATH = "./results/"

    # Source dataset    
    source_dataset = AMAZON_PA.DATASET
    
    # Classification training type (no domain adaptation)
    training_type = SharedParameters.TRAINING_TYPE_CLASSIFICATION
    
    # Part of Checkpoint Directory name
    checkpoint_dir = "checkpoint_tr_" + source_dataset + "_"
    
    # Part of Results Directory name
    results_dir = "results_tr_" + source_dataset + "_"
    
    # Number of runs to do
    runs = "2"
    
    # 2 classes (Deforestation / No Deforastation)
    num_classes = "2"
    
    # Warmup parameter
    warmup = "1"
    
    # Target dataset
    target_dataset = AMAZON_PA.DATASET
    
    # Localization
    DR_LOCALIZATION = ['55']
    
    # Method (architecture) 
    METHODS  = [SharedParameters.METHOD]
    
    # Domain adaptation type
    DA_TYPES = ['None']
    
    # Datasets used for evaluation
    DATASETS = [AMAZON_PA.DATASET, AMAZON_RO.DATASET, CERRADO_MA.DATASET]
    
    # Text to run
    Schedule = []

    for dr_localization in DR_LOCALIZATION:
        for method in METHODS:
            for da in DA_TYPES:
    
                checkpoint_dir_param = checkpoint_dir + training_type + "_" + target_dataset
    
                if args.train:
                    
                    Schedule.append("python " + Train_MAIN_COMMAND + " "
                                    "--classifier_type " + method + " "
                                    "--domain_regressor_type FC "
                                    "--DR_Localization " + dr_localization + " "
                                    "--skip_connections " + SharedParameters.SKIP_CONNECTIONS + " "
                                    "--epochs 100 "
                                    "--batch_size " + SharedParameters.TRAINING_BATCH_SIZE + " "
                                    "--lr " + SharedParameters.LR + " "
                                    "--beta1 0.9 "
                                    "--data_augmentation True "                                                              
                                    "--fixed_tiles True "
                                    "--defined_before False "
                                    "--image_channels 7 "
                                    "--patches_dimension 64 "
                                    "--overlap_s 0.9 "
                                    "--overlap_t 0.9 "
                                    "--compute_ndvi False "
                                    "--balanced_tr False "
                                    "--buffer True "                                
                                    "--porcent_of_last_reference_in_actual_reference 100 "
                                    "--porcent_of_positive_pixels_in_actual_reference_s 2 "
                                    "--porcent_of_positive_pixels_in_actual_reference_t 2 "
                                    "--num_classes " + num_classes + " "                                
                                    "--phase train "
                                    "--training_type " + training_type + " "
                                    "--da_type " + da + " "
                                    "--runs " + runs + " "
                                    "--warmup " + warmup + " "
                                    "--patience " + SharedParameters.PATIENCE + " "
                                    "--checkpoint_dir " + checkpoint_dir_param + " "
                                    "--source_dataset " + source_dataset + " "
                                    "--target_dataset " + target_dataset + " "
                                    "--checkpoint_results_main_path " + Checkpoint_Results_MAIN_PATH + " ")
    
                for target_ds in DATASETS:
    
                    results_dir_param = results_dir + training_type + "_S_" + source_dataset + "_T_" + target_ds
    
                    if args.test:                    
                        Schedule.append("python " + Test_MAIN_COMMAND + " "
                                    "--classifier_type " + method + " "
                                    "--domain_regressor_type FC "
                                    "--DR_Localization " + dr_localization + " "
                                    "--skip_connections " + SharedParameters.SKIP_CONNECTIONS + " "
                                    "--batch_size " + SharedParameters.TESTING_BATCH_SIZE + " "                                
                                    "--overlap 0.75 "
                                    "--image_channels 7 "
                                    "--beta1 0.9 "
                                    "--patches_dimension 64 "
                                    "--compute_ndvi False "
                                    "--num_classes " + num_classes + " "                                
                                    "--phase test "
                                    "--training_type " + training_type + " "
                                    "--da_type " + da + " "
                                    "--checkpoint_dir " + checkpoint_dir_param + " "
                                    "--results_dir " + results_dir_param + " "
                                    "--dataset " + target_ds + " "                                                           
                                    "--checkpoint_results_main_path " + Checkpoint_Results_MAIN_PATH + " ")                
                    if args.metrics:
                        Schedule.append("python " + Metrics_05_MAIN_COMMAND + " "
                                    "--classifier_type " + method + " "
                                    "--domain_regressor_type FC "
                                    "--skip_connections " + SharedParameters.SKIP_CONNECTIONS + " "                                
                                    "--patches_dimension 64 "
                                    "--fixed_tiles True "
                                    "--overlap 0.75 "
                                    "--image_channels 7 "
                                    "--buffer True "                                
                                    "--eliminate_regions True " 
                                    "--Npoints 100 "
                                    "--compute_ndvi False "
                                    "--phase compute_metrics "
                                    "--training_type " + training_type + " "
                                    "--save_result_text True "
                                    "--checkpoint_dir " + checkpoint_dir_param + " "
                                    "--results_dir " + results_dir_param + " "
                                    "--dataset " + target_ds + " "                                
                                    "--checkpoint_results_main_path " + Checkpoint_Results_MAIN_PATH + " ")
                    if args.metrics_avg:
                        Schedule.append("python " + Metrics_th_MAIN_COMMAND + " "
                                    "--classifier_type " + method + " "
                                    "--domain_regressor_type FC "
                                    "--skip_connections " + SharedParameters.SKIP_CONNECTIONS + " "                                
                                    "--patches_dimension 64 "
                                    "--fixed_tiles True "
                                    "--overlap 0.75 "
                                    "--image_channels 7 "
                                    "--buffer True "                                
                                    "--eliminate_regions True "                                
                                    "--Npoints 100 "
                                    "--compute_ndvi False "
                                    "--phase compute_metrics "
                                    "--training_type " + training_type + " "
                                    "--save_result_text False "
                                    "--checkpoint_dir " + checkpoint_dir_param + " "
                                    "--results_dir " + results_dir_param + " "
                                    "--dataset " + target_ds + " "                                
                                    "--checkpoint_results_main_path " + Checkpoint_Results_MAIN_PATH + " ")

    # Execute scripts
    for i in range(len(Schedule)):
        # Argv Arguments    
        argv = Schedule[i].split(' ')[1:-1]
        
        # Execute Respective Script
        if argv[0] == Train_MAIN_COMMAND:
            Main_Train_FC114.main(argv=argv)
        elif argv[0] == Test_MAIN_COMMAND:
            Main_Test_FC114.main(argv=argv)
        elif argv[0] == Metrics_05_MAIN_COMMAND:
            Main_Compute_Metrics_05.main(argv=argv)
        elif argv[0] == Metrics_th_MAIN_COMMAND:
            Main_Compute_Average_Metrics_MT.main(argv=argv)
    
# %% Call main if used as script

if __name__ == '__main__':
    main()