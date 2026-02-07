# %% Import libraries

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import Charts
from skimage.morphology import square, disk

from Tools import *
from Models_FC114 import *
from Amazonia_Legal_RO import AMAZON_RO
from Amazonia_Legal_PA import AMAZON_PA
from Cerrado_Biome_MA import CERRADO_MA

# %% Create parser with function

# An args list can be passed optionally
def parse_args(args=None):
    parser = argparse.ArgumentParser(description='')

    # Defining the meta-parameters
    
    """ Model """
    parser.add_argument('--classifier_type', dest='classifier_type', type=str, default='DeepLab', help='method that will be used, could be used also (siamese_network)')
    parser.add_argument('--domain_regressor_type', dest='domain_regressor_type', type=str, default='FC', help='Architecture of Domain regressor. Values:FC|CONV')
    parser.add_argument('--skip_connections', dest='skip_connections', type=eval, choices=[True, False], default=False, help='method that will be used, could be used also (siamese_network)')
    parser.add_argument('--vertical_blocks', dest='vertical_blocks', type=int, default=10, help='number of blocks which will divide the image vertically')
    parser.add_argument('--horizontal_blocks', dest='horizontal_blocks', type=int, default=10, help='number of blocks which will divide the image horizontally')
    parser.add_argument('--patches_dimension', dest='patches_dimension', type=int, default=29, help= 'dimension of the extracted patches')
    parser.add_argument('--fixed_tiles', dest='fixed_tiles', type=eval, choices=[True, False], default=True, help='decide if tiles will be choosen randomly or not')
    parser.add_argument('--defined_before', dest='defined_before', type=eval, choices=[True, False], default=False, help='decide if tiles will be choosen randomly or not')
    parser.add_argument('--overlap', dest='overlap', type=float, default= 0.75, help= 'stride cadence')
    parser.add_argument('--image_channels', dest='image_channels', type=int, default=7, help='number of image channels')
    parser.add_argument('--buffer', dest='buffer', type=eval, choices=[True, False], default=True, help='Decide wether a buffer around deforestated regions will be performed')
    parser.add_argument('--eliminate_regions', dest='eliminate_regions', type=eval, choices=[True, False], default=True, help='Decide if small regions will be taken into account')
    parser.add_argument('--area_avoided', dest='area_avoided', type=int, default=69, help='area threshold that will be avoided')
    parser.add_argument('--compute_ndvi', dest='compute_ndvi', type=eval, choices=[True, False], default=False, help='Cumpute and stack the ndvi index to the rest of bands')
    parser.add_argument('--phase', dest='phase', default='compute_metrics', help='train, test, compute_metrics')
    parser.add_argument('--training_type', dest='training_type', type=str, default='classification', help='classification|domain_adaptation')
    parser.add_argument('--save_result_text', dest='save_result_text', type=eval, choices=[True, False], default = True, help='decide if a text file results is saved')
    parser.add_argument('--Npoints', dest='Npoints', type=float, default=50, help='Number of thresholds used to compute the curves')

    # Checkpoint dir
    parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./DA_prove_1', help='Domain adaptation checkpoints')

    # Results dir
    parser.add_argument('--results_dir', dest='results_dir', type=str, default='./results_DA_prove_1/', help='results will be saved here')

    """ Images dirs and names """
    parser.add_argument('--dataset', dest='dataset', type=str, default='Amazonia_Legal/',help='The name of the dataset used')
    parser.add_argument('--images_section', dest='images_section', type=str, default='Organized/Images/', help='Folder for the images')
    parser.add_argument('--reference_section', dest='reference_section', type=str, default='Organized/References/', help='Folder for the reference')
    parser.add_argument('--data_type', dest='data_type', type=str, default='.npy', help= 'Type of the input images and references')
    parser.add_argument('--data_t1_year', dest='data_t1_year', type=str, default='2016', help='Year of the time 1 image')
    parser.add_argument('--data_t2_year', dest='data_t2_year', type=str, default='2017', help='Year of the time 2 image')
    parser.add_argument('--data_t1_name', dest='data_t1_name', type=str, default='18_07_2016_image', help='image 1 name')
    parser.add_argument('--data_t2_name', dest='data_t2_name', type=str, default='21_07_2017_image', help='image 2 name')
    parser.add_argument('--reference_t1_name', dest='reference_t1_name', type=str, default='PAST_REFERENCE_FOR_2017_EPSG32620', help='reference 1 name')
    parser.add_argument('--reference_t2_name', dest='reference_t2_name', type=str, default=None, help='reference 2 name')

    """ Main paths """
    parser.add_argument('--dataset_main_path', dest='dataset_main_path', type=str, default=None, help='Dataset main path')
    parser.add_argument('--checkpoint_results_main_path', dest='checkpoint_results_main_path', type=str, default=None)
    # parser.add_argument('--uncertainty_results_main_path', dest='uncertainty_results_main_path', type=str, default=None)

    return parser.parse_args(args=args)

# %% Main function to execute

def main(argv=None):
    # Command line argv is passed if not specified
    if argv is None:
        argv = sys.argv
    
    # Parse args
    args = parse_args(args=argv[1:])

    # Print args
    print("Module name: ", __name__)
    print("Args: ")
    print(json.dumps(vars(args), indent=4), end='\n\n')
    

    Thresholds = np.array([0.5])
    
    if args.dataset == AMAZON_RO.DATASET:        
        dataset = AMAZON_RO(args)
    elif args.dataset ==  AMAZON_PA.DATASET:        
        dataset = AMAZON_PA(args)
    elif args.dataset == CERRADO_MA.DATASET:        
        dataset = CERRADO_MA(args)
    else:
        raise Exception("Invalid dataset argument: " + args.dataset)

    args.area_avoided = int(dataset.AREA_AVOIDED)
    args.horizontal_blocks = int(dataset.HORIZONTAL_BLOCKS)
    args.vertical_blocks = int(dataset.VERTICAL_BLOCKS)

    args.results_dir = args.checkpoint_results_main_path + 'results/' + args.results_dir + '/'
    args.checkpoint_dir = args.checkpoint_results_main_path + 'checkpoints/' + args.checkpoint_dir + '/'

    if os.path.exists(os.path.join(args.results_dir, "Results.txt")):
        os.remove(os.path.join(args.results_dir, "Results.txt"))

    initial_flag = True

    counter = 0
    files = os.listdir(args.results_dir)
    for i in range(0, len(files)):
        Hit_map_path = args.results_dir + files[i] + '/hit_map.npy'
        args.file = files[i]
        if os.path.exists(Hit_map_path):
            hit_map = np.load(Hit_map_path)

            if initial_flag:
                HIT_MAP = np.zeros_like(hit_map)
                initial_flag = False
            HIT_MAP += hit_map

            fields_file = files[i].split('_')
            checkpoint_name = fields_file[0] + '_' + fields_file[3] + '_' + fields_file[1] + '_' + fields_file[4] + '_' + fields_file[5] + '_' + fields_file[6] + '_' + fields_file[7] + '_'+ fields_file[8] + '_' + fields_file[9] + '_' + fields_file[10] + '_' + fields_file[11]
            args.save_checkpoint_path = args.checkpoint_dir + '/' + checkpoint_name + '/'
            #need to put the path of the checkpoint to recover if needed the original train, validation, and test tiles.
            dataset.Tiles_Configuration(args, i)

            if args.save_result_text:
                # Open a file in order to save the training history
                # 'w' parameter - Write mode - will overwrite any existing content
                f = open(args.results_dir + "/Results.txt","a")

                if counter == 0:
                    ACCURACY_ = []
                    FSCORE_ = []
                    RECALL_ = []
                    PRECISION_ = []
                    ALERT_AREA_ = []
                    mAP_ = []

            args.create_classification_map = False

            ACCURACY, FSCORE, RECALL, PRECISION, _, ALERT_AREA, _, _, _, _, _ = Metrics_For_Test(hit_map, None, dataset.references[0], dataset.references[1],
                                                                                                 dataset.Train_tiles, dataset.Valid_tiles, dataset.Undesired_tiles,
                                                                                                 Thresholds, None,
                                                                                                args)

            _, _, RECALL_for_mAP, PRECISION_for_mAP, _, _, _, _, _, _, _  = Metrics_For_Test(hit_map, None, dataset.references[0], 
                                                                             dataset.references[1],dataset.Train_tiles, dataset.Valid_tiles, dataset.Undesired_tiles,
                                                                            None, None, args)
            
            _,_,mAP = Charts.computeMap(100, RECALL_for_mAP, PRECISION_for_mAP)

            if args.save_result_text:
                ACCURACY_.append(ACCURACY[0,0])
                FSCORE_.append(FSCORE[0,0])
                RECALL_.append(RECALL[0,0])
                PRECISION_.append(PRECISION[0,0])
                ALERT_AREA_.append(ALERT_AREA[0,0])
                mAP_.append(mAP)
                
                f.write("Run: %d Accuracy: %.2f%% F1-Score: %.2f%% Recall: %.2f%% Precision: %.2f%% Area: %.2f%% mAP: %.2f%% File Name: %s\n" % (counter, ACCURACY[0,0], FSCORE[0,0], RECALL[0,0], PRECISION[0,0], ALERT_AREA[0,0], mAP, args.file))
                f.close()
            else:
                print('Coming up!')
                
            counter += 1   

    if args.save_result_text:
        f = open(args.results_dir + "/Results.txt","a")
        ACCURACY_m = np.mean(ACCURACY_)
        FSCORE_m = np.mean(FSCORE_)
        RECALL_m = np.mean(RECALL_)
        PRECISION_m = np.mean(PRECISION_)
        ALERT_AREA_m = np.mean(ALERT_AREA_)
        mAP_m = np.mean(mAP_)


        ACCURACY_s = np.std(ACCURACY_)
        FSCORE_s = np.std(FSCORE_)
        RECALL_s = np.std(RECALL_)
        PRECISION_s = np.std(PRECISION_)
        ALERT_AREA_s = np.std(ALERT_AREA_)
        mAP_s = np.std(mAP_)

        #histories.sendLoss(loss = FSCORE_m, epoch = i + 1, total_epochs = len(files) + 1)
        f.write("Mean: %d Accuracy: %f%% F1-Score: %f%% Recall: %f%% Precision: %f%% Area: %f%% mAP: %f%% \n" % ( 0, ACCURACY_m, FSCORE_m, RECALL_m, PRECISION_m, ALERT_AREA_m, mAP_m))
        f.write("Std: %d Accuracy: %.2f%% F1-Score: %.2f%% Recall: %.2f%% Precision: %.2f%% Area: %.2f%% mAP: %f%% \n" % ( 0, ACCURACY_s, FSCORE_s, RECALL_s, PRECISION_s, ALERT_AREA_s, mAP_s))
        f.close()

# %% Call main if used as script

if __name__=='__main__':
    main()
