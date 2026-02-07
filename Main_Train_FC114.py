# %% Import libraries

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime
from SharedParameters import TRAINING_TYPE_DOMAIN_ADAPTATION,TRAINING_TYPE_CLASSIFICATION,ROLE_SOURCE,ROLE_TARGET
from Tools import cleanup_folder
import time

#from tensordash.tensordash import Tensordash, Customdash

from Tools import *
from Models_FC114 import *
from Amazonia_Legal_RO import AMAZON_RO
from Amazonia_Legal_PA import AMAZON_PA
from Cerrado_Biome_MA import CERRADO_MA

# %% Create parser with function

# An args list can be passed optionally
def parse_args(args=None):    
    parser = argparse.ArgumentParser(description='')
    
    # Defining the meta-paramerts
    
    """ Model """
    # Model name
    parser.add_argument('--classifier_type', dest='classifier_type', type=str, default='DeepLab', help='method that will be used, could be used also (siamese_network)')
    # Skip connections
    parser.add_argument('--skip_connections', dest='skip_connections', type=eval, choices=[True, False], default=False, help='method that will be used, could be used also (siamese_network)')
    # Domain Regressor Type
    parser.add_argument('--domain_regressor_type', dest='domain_regressor_type', type=str, default='FC', help='Architecture of Domain regressor. Values:FC|CONV')
    # Localization of Regressor 
    parser.add_argument('--DR_Localization', dest='DR_Localization', type=int, default=-1, help='The layer in whic the Domain regressor will act')
    
    """ Training parameters """
    # Epochs
    parser.add_argument('--epochs', dest='epochs', type=int, default=100, help='number of epochs')
    # Batch size
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='number images in batch')
    # Optimizer hyperparameters
    parser.add_argument('--lr', dest='lr', type=float, default=0.01, help='initial learning rate for adam')
    parser.add_argument('--gamma', dest='gamma', type=float, default=2.5, help='parameter for lambda computation')
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.9, help='momentum term of adam')
    # Image processing hyperparameters
    parser.add_argument('--data_augmentation', dest='data_augmentation', type=eval, choices=[True, False], default=True, help='if data argumentation is applied to the data')
    
    """ Extraction parameters """
    # TODO LUCAS:Em quantas colunas ou linhas eu irei dividir minha imagem para gerar os quadradinhos (patches)
    parser.add_argument('--source_vertical_blocks', dest='source_vertical_blocks', type=int, default=10, help='number of blocks which will divide the image vertically')
    parser.add_argument('--source_horizontal_blocks', dest='source_horizontal_blocks', type=int, default=10, help='number of blocks which will divide the image horizontally')
    parser.add_argument('--target_vertical_blocks', dest='target_vertical_blocks', type=int, default=10, help='number of blocks which will divide the image vertically')
    parser.add_argument('--target_horizontal_blocks', dest='target_horizontal_blocks', type=int, default=10, help='number of blocks which will divide the image horizontally')
    
    parser.add_argument('--fixed_tiles', dest='fixed_tiles', type=eval, choices=[True, False], default=True, help='decide if tiles will be choosen randomly or not')
    parser.add_argument('--defined_before', dest='defined_before', type=eval, choices=[True, False], default=False, help='decide if tiles will be choosen randomly or not')
    parser.add_argument('--image_channels', dest='image_channels', type=int, default=7, help='number of image channels')
    parser.add_argument('--patches_dimension', dest='patches_dimension', type=int, default=128, help= 'dimension of the extracted patches')
    
    # Patch overlap in source and target
    parser.add_argument('--overlap_s', dest='overlap_s', type=float, default= 0.75, help= 'stride cadence')
    parser.add_argument('--overlap_t', dest='overlap_t', type=float, default= 0.75, help= 'stride cadence')
    
    # compute ndvi refere-se a um indice. Era algum tipo de stack de bandas. compute_ndvi = False. Pode ignorar e manter assim.
    parser.add_argument('--compute_ndvi', dest='compute_ndvi', type=eval, choices=[True, False], default=False, help='Compute and stack the ndvi index to the rest of bands')
    parser.add_argument('--balanced_tr', dest='balanced_tr', type=eval, choices=[True, False], default=False, help='Decide wether a balanced training will be performed')
    #parser.add_argument('--balanced_vl', dest='balanced_vl', type=eval, choices=[True, False], default=True, help='Decide wether a balanced training will be performed')
    
    # TODO LUCAS:Parâmetro buffer para quando for converter de imagem vetorial para pixel
    parser.add_argument('--buffer', dest='buffer', type=eval, choices=[True, False], default=True, help='Decide wether a buffer around deforestated regions will be performed')
    
    parser.add_argument('--porcent_of_last_reference_in_actual_reference', dest='porcent_of_last_reference_in_actual_reference', type=int, default=100, help='Porcent of number of pixels of last reference in the actual reference')
    parser.add_argument('--porcent_of_positive_pixels_in_actual_reference_s', dest='porcent_of_positive_pixels_in_actual_reference_s', type=int, default=2, help='Porcent of number of pixels of last reference in the actual reference in source domain')
    parser.add_argument('--porcent_of_positive_pixels_in_actual_reference_t', dest='porcent_of_positive_pixels_in_actual_reference_t', type=int, default=2, help='Porcent of number of pixels of last reference in the actual reference in target domain')
    parser.add_argument('--num_classes', dest='num_classes', type=int, default=2, help='Number of classes comprised in both domains (classification)')
    
    """ Other training parameters """
    # Phase
    parser.add_argument('--phase', dest='phase', default='train', help='train, test, generate_image, create_dataset')
    parser.add_argument('--training_type', dest='training_type', type=str, default='classification', help='classification|domain_adaptation|domain_adaptation_check')
    parser.add_argument('--da_type', dest='da_type', type=str, default='DR', help='CL|DR|CL_DR')
    
    # TODO LUCAS:Geralmente rodamos 10x
    parser.add_argument('--runs', dest='runs', type=int, default=1, help='number of executions of the algorithm')
    #parser.add_argument('--scatter_plot', dest='scatter_plot', type=eval, choices=[True, False], default=True, help='Decide if a scatter plot is done during the training')
    #parser.add_argument('--change_every_epoch', dest='change_every_epoch', type=eval, choices=[True, False], default=False, help='Decide if the target set will be change every epoch in order to balance the training')
    
    # Early stop parameter
    parser.add_argument('--patience', dest='patience', type=int, default=10, help='number of epochs without improvement to apply early stop')
    parser.add_argument('--warmup', dest='warmup', type=int, default=1, help='number of epochs without backpropagation of discriminator gradients')
    
    # Checkpoint dir
    parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='prove', help='Domain adaptation checkpoints')
    
    # Images dir and names
    parser.add_argument('--source_dataset', dest='source_dataset', type=str, default='Amazon_RO',help='The name of the dataset used')
    parser.add_argument('--target_dataset', dest='target_dataset', type=str, default='Cerrado_MA',help='The name of the dataset used')
    parser.add_argument('--images_section', dest='images_section', type=str, default='Organized/Images/', help='Folder for the images')
    parser.add_argument('--reference_section', dest='reference_section', type=str, default='Organized/References/', help='Folder for the reference')
    parser.add_argument('--data_type', dest='data_type', type=str, default='.npy', help= 'Type of the input images and references')
    
    parser.add_argument('--target_data_t1_year', dest='target_data_t1_year', type=str, default='2017', help='Year of the time 3 image')
    parser.add_argument('--target_data_t2_year', dest='target_data_t2_year', type=str, default='2018', help='Year of the time 4 image')
    parser.add_argument('--source_data_t1_name', dest='source_data_t1_name', type=str, default=None, help='image 1 name')
    parser.add_argument('--source_data_t2_name', dest='source_data_t2_name', type=str, default=None, help='image 2 name')
    parser.add_argument('--target_data_t1_name', dest='target_data_t1_name', type=str, default=None, help='image 3 name')
    parser.add_argument('--target_data_t2_name', dest='target_data_t2_name', type=str, default=None, help='image 4 name')
    parser.add_argument('--source_reference_t1_name', dest='source_reference_t1_name', type=str, default=None, help='reference 1 name')
    parser.add_argument('--source_reference_t2_name', dest='source_reference_t2_name', type=str, default=None, help='reference 2 name')
    parser.add_argument('--target_reference_t1_name', dest='target_reference_t1_name', type=str, default=None, help='reference 1 name')
    parser.add_argument('--target_reference_t2_name', dest='target_reference_t2_name', type=str, default=None, help='reference 2 name')
    
    parser.add_argument('--dataset_main_path', dest='dataset_main_path', type=str, default='/code/Datasets/', help='Dataset main path')
    parser.add_argument('--checkpoint_results_main_path', dest='checkpoint_results_main_path', type=str, default=None)
    parser.add_argument('--save_intermediate_model', dest='save_intermediate_model',type=eval, choices=[True, False], default=True, help='Save intermediate models or not')
    
    parser.add_argument('--source_targets_balanced', dest='source_targets_balanced', type=eval, choices=[True, False], default=True, help='Applies for Multi-target training. Decides whether each one of source and target datasets will correspont to 1/3 of training data. If not, source will correspond 50%% and both target datasets will share another 50%%.')
    
    parser.add_argument('--discriminate_domain_targets', dest='discriminate_domain_targets', type=eval, choices=[True, False], default=False, help='Applies for Multi-target training. Decides whether each target dataset will be assigned a different domain label or every target dataset will get the same label.')
    
    parser.add_argument('--num_domains', dest='num_domains', type=int, default=None, help='Number of classes for discriminator training (domain adaptation)')
    
    # parser.add_argument('--uncertainty_results_main_path', dest='uncertainty_results_main_path', type=str, default=None)

    return parser.parse_args(args=args)


# %% Main function to execute

def main(argv=None):
    # custom argv can be passed, but system argv is used if argv isn't specified
    if argv is None:
        argv = sys.argv
        
    args = parse_args(args=argv[1:])
    
    # Print args passed
    print("Module name: ", __name__)
    print("Args: ")
    print(json.dumps(vars(args), indent=4), end='\n\n')
   
    # Checkpoint Directories
    print("Creating main Checkpoint dir: ", args.checkpoint_results_main_path + 'checkpoints/')
    if not os.path.exists(args.checkpoint_results_main_path + 'checkpoints/'):
        os.makedirs(args.checkpoint_results_main_path + 'checkpoints/')

    print("Current Checkpoint dir: ", args.checkpoint_results_main_path + 
          'checkpoints/' + args.checkpoint_dir, end='\n\n')
    args.checkpoint_dir = args.checkpoint_results_main_path + 'checkpoints/' + args.checkpoint_dir

    # Source dataset(s) and Target dataset(s)
    dataset_s = []
    dataset_t = []

    print("Building list of source dataset(s) with: ", args.source_dataset)
    if AMAZON_RO.DATASET in args.source_dataset:
        args.role = ROLE_SOURCE      
        args.reference_t2_name = args.source_reference_t2_name  
        dataset_s.append(AMAZON_RO(args)) # AMAZON_RO Instance
    if AMAZON_PA.DATASET in args.source_dataset:
        args.role = ROLE_SOURCE
        args.reference_t2_name = args.source_reference_t2_name
        dataset_s.append(AMAZON_PA(args))
    if CERRADO_MA.DATASET in args.source_dataset:     
        args.role = ROLE_SOURCE   
        args.reference_t2_name = args.source_reference_t2_name
        dataset_s.append(CERRADO_MA(args))
    print() # blank line separator

    print("Building list of target dataset(s) with: ", args.target_dataset)
    if AMAZON_RO.DATASET in args.target_dataset:              
        args.role = ROLE_TARGET     
        args.reference_t2_name = args.target_reference_t2_name
        dataset_t.append(AMAZON_RO(args))

    if AMAZON_PA.DATASET in args.target_dataset: 
        args.role = ROLE_TARGET               
        args.reference_t2_name = args.target_reference_t2_name
        dataset_t.append(AMAZON_PA(args))

    if CERRADO_MA.DATASET in args.target_dataset:
        args.role = ROLE_TARGET
        args.reference_t2_name = args.target_reference_t2_name
        dataset_t.append(CERRADO_MA(args))
    print() # blank line separator

    # The training is multi-source or multi-target, not both
    if len(dataset_s) > 1 and len(dataset_t) > 1:
        raise Exception("Error. Both source and target datasets cannot have multiple domains.")
    
    # Dataset cannot be empty
    if len(dataset_s) == 0 or len(dataset_t) == 0:
        raise Exception("Error. Empty dataset.")

    # Clean up checkpoint dir
    print(f'Cleaning up {args.checkpoint_dir}', end='/n/n')
    cleanup_folder(args.checkpoint_dir)
    
    # Run Loop
    for i in range(args.runs):
        print('[*]Training Run %d'%(i))
        dataset = []
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        print(dt_string)
        if args.training_type == TRAINING_TYPE_CLASSIFICATION:
            args.save_checkpoint_path = os.path.join(args.checkpoint_dir, args.classifier_type + '_' + dt_string)
        elif args.training_type == TRAINING_TYPE_DOMAIN_ADAPTATION:
            args.save_checkpoint_path = os.path.join(args.checkpoint_dir, 'Tr_M_' + dt_string)

        args.num_run = str(i)
        
        if not os.path.exists(args.save_checkpoint_path):
            os.makedirs(args.save_checkpoint_path)
            
        # Writing the args into a file
        with open(os.path.join(args.save_checkpoint_path,'commandline_args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        
        
        args.overlap = args.overlap_s
        args.porcent_of_positive_pixels_in_actual_reference = args.porcent_of_positive_pixels_in_actual_reference_s
        
        for s in dataset_s:
            s.Tiles_Configuration(args, i)
            s.Coordinates_Creator(args, i)
        
        args.overlap = args.overlap_t
        args.porcent_of_positive_pixels_in_actual_reference = args.porcent_of_positive_pixels_in_actual_reference_t

        for t in dataset_t:
            t.Tiles_Configuration(args, i)
            t.Coordinates_Creator(args, i)

        print('[*]Initializing the model...')
        model = Models(args, dataset_s, dataset_t)

        model.Train()

        time.sleep(300)
        
# %% Call main if used as script

if __name__=='__main__':
    main()