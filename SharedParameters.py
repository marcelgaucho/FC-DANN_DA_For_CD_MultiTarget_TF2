# %% Import Libraries

import os

# %% Define HOME

home = os.getcwd()
home = home + '/../..' # Home is parent dir from the workspace directory

# %% Other parameters

Dataset_MAIN_PATH = str(home)+"/workspace/dataset/"
METHOD = "DeepLab"

Train_MAIN_COMMAND = "Main_Train_FC114.py"
Test_MAIN_COMMAND = "Main_Test_FC114.py"
Metrics_05_MAIN_COMMAND = "Main_Compute_Metrics_05.py"
Metrics_th_MAIN_COMMAND = "Main_Compute_Average_Metrics_MT.py"

AVG_MAIN_PATH = "./results/results_avg/"

RESULTS_MAIN_PATH = "./results/results/"

CHECKPOINTS_MAIN_PATH = "./results/checkpoints/"

ROLE_SOURCE = "S"
ROLE_TARGET = "T"

PHASE_TRAIN = "train"
PHASE_TEST = "test"
PHASE_METRICS = "compute_metrics"

LR = str(0.0001)


#2.5
GAMMA = str(2.5)

PATIENCE = str(10)

TRAINING_BATCH_SIZE = "32"
TESTING_BATCH_SIZE = "500"

Full_Path_Train_MAIN_COMMAND = "$HOME/workspace/FC-DANN_DA_For_CD_MultiTarget_TF2/"+Train_MAIN_COMMAND
Full_Path_Test_MAIN_COMMAND = "$HOME/workspace/FC-DANN_DA_For_CD_MultiTarget_TF2/"+Test_MAIN_COMMAND
Full_Path_Metrics_05_MAIN_COMMAND = "$HOME/workspace/FC-DANN_DA_For_CD_MultiTarget_TF2/"+Metrics_05_MAIN_COMMAND
Full_Path_Metrics_th_MAIN_COMMAND = "$HOME/workspace/FC-DANN_DA_For_CD_MultiTarget_TF2/"+Metrics_th_MAIN_COMMAND

IMAGES_SECTION = "Organized/Images/"
REFERENCE_SECTION = "Organized/References/"
T1_YEAR = "2016"
T2_YEAR = "2017"
DATA_TYPE = ".npy"

BUFFER_DIMENSION_IN = 0
BUFFER_DIMENSION_OUT = 2

SKIP_CONNECTIONS = str(True)

TRAINING_TYPE_CLASSIFICATION = "classification"
TRAINING_TYPE_DOMAIN_ADAPTATION = "domain_adaptation"

DMDA_FILE_TITLE = 'DMDA'
DMDA_SOURCE_FILE_TITLE = 'DMDA_SOURCE'
DA_MULTI_SOURCE_FILE_TITLE = 'Multi_source'
DA_MULTI_TARGET_FILE_TITLE = 'Multi_target'
DA_SINGLE_TARGET_FILE_TITLE = 'Single_target'
DA_MULTI_SOURCE_TITLE = 'DMDA Multi-Source '
DA_MULTI_TARGET_TITLE = 'DMDA Multi-Target '
DA_SINGLE_TARGET_TITLE = 'DA Single-Target '
DA_CHART_TITLE = 'DMDA Multi-Target experiments comparison'

#LOWER_BOUND_LABEL = 'Source only training\\\(lowerbound)'
#UPPER_BOUND_SOURCE_ONLY_LABEL = 'Training on target\\\(upperbound)'
UPPER_BOUND_SOURCE_ONLY_LABEL = 'Training on target$^{1}$'
LOWER_BOUND_LABEL = 'Source only training$^{2}$'
SINGLE_TARGET_LABEL = 'DA Single-target$^{3}$'
MULTI_TARGET_LABEL = 'DMDA Multi-target$^{4}$'
MULTI_SOURCE_LABEL = 'DMDA Multi-source$^{5}$'
MULTI_SOURCE_LABEL_NO_DA = 'Multi-source only training'

'''
FORMAT_CHART_TITLE =                    'Training on {} | Validating on {}'
FORMAT_LOWER_BOUND_LABEL =              'Tr({})|Val({})\nSource only training'
FORMAT_UPPER_BOUND_SOURCE_ONLY_LABEL =  'Tr({})|Val({})\nTraining on target'
FORMAT_SINGLE_TARGET_LABEL =            'Tr({})|Val({}→{})\nDA Single-Target'
FORMAT_MULTI_TARGET_LABEL =             'Tr({})|Val({},{}→{})\nDA Multi-Target'
FORMAT_MULTI_SOURCE_LABEL =             'Tr({},{})|Val({}→{},{})\nDA Multi-Source'
FORMAT_MULTI_SOURCE_NO_DA_LABEL =       'Tr({},{})|Val({})\nMulti-Source only training'
'''

FORMAT_F1_SOURCE_CHART_TITLE =          'Source {}'
FORMAT_F1_CHART_TITLE =                 'Target {}'
FORMAT_CHART_TITLE =                    'Source {} | Target {}'
FORMAT_LOWER_BOUND_LABEL =              'Source {} | Target {}\nSource only training'
FORMAT_UPPER_BOUND_SOURCE_ONLY_LABEL =  'Source {} | Target {}\nTraining on target'
FORMAT_SINGLE_TARGET_LABEL =            'Source {} | Target {}\nDA Single-target'
FORMAT_MULTI_TARGET_LABEL =             'Source {} | Target {},{} | Test {}\nDMDA Multi-target'
FORMAT_MULTI_SOURCE_LABEL =             'Source {},{} | Target {}\nDMDA Multi-source'
FORMAT_MULTI_SOURCE_NO_DA_LABEL =       'Source {},{} | Target {}\nMulti-source only training'

formatted_f1_source_chart_title = lambda x: FORMAT_F1_SOURCE_CHART_TITLE.format(x)
formatted_f1_chart_title = lambda x: FORMAT_F1_CHART_TITLE.format(x)
formatted_chart_title = lambda x, y: FORMAT_CHART_TITLE.format(x, y)
formatted_lower_bound_label = lambda x, y: FORMAT_LOWER_BOUND_LABEL.format(x, y)
formatted_upper_bound_source_only_label = lambda y: FORMAT_UPPER_BOUND_SOURCE_ONLY_LABEL.format(y,y)
formatted_single_target_label = lambda x, y: FORMAT_SINGLE_TARGET_LABEL.format(x, y, x)
formatted_multi_target_label = lambda x, y1, y2, y3: FORMAT_MULTI_TARGET_LABEL.format(x, y1, y2, y3)
formatted_multi_source_label = lambda x, y, z: FORMAT_MULTI_SOURCE_LABEL.format(x, y, z, x, y)
formatted_multi_source_no_da_label = lambda x, y, z: FORMAT_MULTI_SOURCE_NO_DA_LABEL.format(x, y, z)

EXPERIMENTS_LABELS = [    
    'Multi-Domain Disc.',
    'Source-Target Disc.'
]

EXPERIMENTS_LABELS_LB = [    
    'Multi-Domain Disc.',
    'Source-Target Disc.'
]

MULTI_SOURCE_EXPERIMENTS_LABELS = [    
    'Source-Target Disc.'
]


INCLUDE_DA_RESULTS = True

DISCRIMINATE_DOMAIN_TARGETS = str(True)