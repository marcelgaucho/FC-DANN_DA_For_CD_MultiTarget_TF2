from osgeo import gdal
import os
import sys
import skimage
import time
import numpy as np
import scipy.io as sio
from tqdm import trange
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from contextlib import redirect_stdout
from keras.optimizers import Adam
from keras.backend import clear_session
from SharedParameters import TRAINING_TYPE_DOMAIN_ADAPTATION,TRAINING_TYPE_CLASSIFICATION, PHASE_TRAIN
from Tools import Data_Augmentation_Definition, Patch_Extraction, Data_Augmentation_Execution, Classification_Maps, compute_metrics, mask_creation
from Networks import *
from discriminators import *
from collections import Counter

import math
from Tools import *
from Networks import *
from flip_gradient import flip_gradient


class Models():
    def __init__(self, args, dataset_s, dataset_t):
        # Disable Tensorflow 2 behavior (Eager Execution)
        tf.compat.v1.disable_v2_behavior()
        
        # Create new graph and set seed to random (time related)
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(int(time.time()))
        
        # Variables used in object
        self.args = args
        self.dataset_s = dataset_s
        self.dataset_t = dataset_t
        self.loss_dr_threshold = 1. # loss threshold for domain regressor
        
        # Set number of domains
        if self.args.discriminate_domain_targets:
            self.loss_dr_threshold = 1.5
            if self.args.phase == PHASE_TRAIN:
                self.num_domains = len(self.dataset_s) + len(self.dataset_t)
            else:
                self.num_domains = 3
        else:
            self.num_domains = 2

        self.args.num_domains = self.num_domains

        print("self.discriminate_domain_targets: " + str(self.args.discriminate_domain_targets))
        print("self.num_domains: " + str(self.num_domains))

        self.segmentation_history = {}
        self.discriminator_history = {}

        self.segmentation_history["loss"] = []        
        self.segmentation_history["f1"] = []
        self.segmentation_history["val_loss"] = []
        self.segmentation_history["val_f1"] = []

        self.discriminator_history["loss"] = []
        self.discriminator_history["val_loss"] = []
        self.discriminator_history["accuracy"] = []
        self.discriminator_history["val_accuracy"] = []

        self.learning_rate = tf.compat.v1.placeholder(tf.float32, [], name="learning_rate")

        if self.args.compute_ndvi:
            self.data = tf.compat.v1.placeholder(tf.float32, [None, self.args.patches_dimension, self.args.patches_dimension, 2 * self.args.image_channels + 2], name = "data")
        else:
            self.data = tf.compat.v1.placeholder(tf.float32, [None, self.args.patches_dimension, self.args.patches_dimension, 2 * self.args.image_channels], name = "data")

        if self.args.domain_regressor_type == 'Dense':
            self.label_d = tf.compat.v1.placeholder(tf.float32, [None, None, None, self.num_domains], name = "label_d")
        if self.args.domain_regressor_type == 'FC':
            self.label_d = tf.compat.v1.placeholder(tf.float32, [None, self.num_domains], name = "label_d")

        self.label_c = tf.compat.v1.placeholder(tf.float32, [None, self.args.patches_dimension, self.args.patches_dimension, self.args.num_classes], name = "label_c")
        self.mask_c = tf.compat.v1.placeholder(tf.float32, [None, self.args.patches_dimension, self.args.patches_dimension], name="labeled_samples")
        self.class_weights = tf.compat.v1.placeholder(tf.float32, [None, self.args.patches_dimension, self.args.patches_dimension, self.args.num_classes], name="class_weights")
        self.L = tf.compat.v1.placeholder(tf.float32, [], name="L" )
        self.phase_train = tf.compat.v1.placeholder(tf.bool, name = "phase_train")

        if self.args.classifier_type == 'Unet':
            self.args.encoder_blocks = 5
            self.args.base_number_of_features = 16
            self.Unet = Unet(self.args)
            #Defining the classifiers

            Encoder_Outputs = self.Unet.build_Unet_Encoder(self.data, name = "Unet_Encoder")
            Decoder_Outputs = self.Unet.build_Unet_Decoder(Encoder_Outputs[-1], Encoder_Outputs, name="Unet_Decoder")

            if self.args.training_type == 'domain_adaptation':
                if self.args.DR_Localization > 1 and self.args.DR_Localization <= len(Encoder_Outputs):
                    self.features_c = Encoder_Outputs[self.args.DR_Localization]
                elif self.args.DR_Localization < 0 and self.args.DR_Localization >= -len(Decoder_Outputs):
                    self.features_c = Decoder_Outputs[self.args.DR_Localization]
                elif self.args.DR_Localization > len(Encoder_Outputs) and self.args.DR_Localization < (len(Encoder_Outputs) + len(Decoder_Outputs)):
                    self.features_c = Decoder_Outputs[self.args.DR_Localization - (len(Encoder_Outputs) + len(Decoder_Outputs))]
                else:
                    self.features_c = Encoder_Outputs[-1]
                    print("Warning: Please select the layer index correctly!")

            self.logits_c = Decoder_Outputs[-2]
            self.prediction_c = Decoder_Outputs[-1]

        elif self.args.classifier_type == 'DeepLab':

            self.args.backbone = 'xception'
            #self.args.filters = (16, 32)
            #self.args.stages = (2, 3)
            self.args.aspp_rates = (1, 2, 3)
            self.args.data_format = 'channel_last'
            self.args.bn_decay = 0.9997

            self.DeepLab = DeepLabV3Plus(self.args)

            #Building the encoder
            Encoder_Outputs, low_Level_Features = self.DeepLab.build_DeepLab_Encoder(self.data, name = "DeepLab_Encoder")
            #Building Decoder
            Decoder_Outputs = self.DeepLab.build_DeepLab_Decoder(Encoder_Outputs[-1], low_Level_Features, name = "DeepLab_Decoder")

            if self.args.training_type == TRAINING_TYPE_DOMAIN_ADAPTATION:
                if self.args.DR_Localization > 1 and self.args.DR_Localization <= len(Encoder_Outputs):
                    self.features_c = Encoder_Outputs[self.args.DR_Localization]
                elif self.args.DR_Localization < 0 and self.args.DR_Localization >= -len(Decoder_Outputs):
                    self.features_c = Decoder_Outputs[self.args.DR_Localization]
                elif self.args.DR_Localization > len(Encoder_Outputs) and self.args.DR_Localization < (len(Encoder_Outputs) + len(Decoder_Outputs)):
                    self.features_c = Decoder_Outputs[self.args.DR_Localization - (len(Encoder_Outputs) + len(Decoder_Outputs))]
                else:
                    print("Please select the layer index correctly!")

            self.logits_c = Decoder_Outputs[-2]
            self.prediction_c = Decoder_Outputs[-1]

        if self.args.training_type == TRAINING_TYPE_DOMAIN_ADAPTATION:
            if 'DR' in self.args.da_type:
                flip_feature = flip_gradient(self.features_c, self.L)
                self.DR = Domain_Regressors(self.args)

                if self.args.domain_regressor_type == 'FC':
                    DR_Ouputs = self.DR.build_Domain_Classifier_Arch(flip_feature, name = 'FC_Domain_Classifier')
                if self.args.domain_regressor_type == 'Dense':
                    DR_Ouputs = self.DR.build_Dense_Domain_Classifier(flip_feature, name = 'Dense_Domain_Classifier')

                self.logits_d = DR_Ouputs[-2]

        if self.args.phase == PHASE_TRAIN:
            
            self.summary(Encoder_Outputs, "Encoder: ")
            self.summary(Decoder_Outputs, "Decoder: ")
            #Defining losses
            # Classifier loss, only for the source labeled samples
            temp_loss = self.weighted_cross_entropy_c(self.label_c, self.prediction_c, self.class_weights)
            
            # Essa mask_c deixa de fora os pixels que eu não me importo. A rede vai gerar um resultado, mas eu não nao me importo com essas saidas
            self.classifier_loss =  tf.reduce_sum(self.mask_c * temp_loss) / tf.reduce_sum(self.mask_c)

            if self.args.training_type == 'classification':
                self.total_loss = self.classifier_loss
            else:
                if 'DR' in self.args.da_type:

                    self.summary(DR_Ouputs, "Domain_Regressor: ")

                    print('Input shape of D')
                    print(np.shape(self.features_c))
                    self.D_out_shape = self.logits_d.get_shape().as_list()[1:]
                    print('Output shape of D')
                    print(self.D_out_shape)

                    self.domainregressor_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.logits_d, labels = tf.stop_gradient( self.label_d)))
                    self.total_loss = self.classifier_loss + self.domainregressor_loss
                else:
                    self.total_loss = self.classifier_loss

            # Defining the Optimizers
            self.training_optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate, self.args.beta1).minimize(self.total_loss) #com learning rate decay
            self.saver = tf.compat.v1.train.Saver(max_to_keep=5)
            self.sess=tf.compat.v1.Session()
            self.sess.run(tf.compat.v1.initialize_all_variables())

        elif self.args.phase == 'test':
            self.dataset = dataset_t
            self.saver = tf.compat.v1.train.Saver(max_to_keep=5)
            self.sess=tf.compat.v1.Session()
            self.sess.run(tf.compat.v1.initialize_all_variables())
            print('[*]Loading the feature extractor and classifier trained models...')
            mod = self.load(self.args.trained_model_path)
            if mod:
                print(" [*] Weights loaded successfuly")
            else:
                print(" [!] Load failed...")
                sys.exit()

    

    def summary(self, net, name):
        summaryFile = os.path.join(self.args.save_checkpoint_path,"Architecture.txt")
        f = open(summaryFile,"a")
        f.write(name + "\n")
        for i in range(len(net)):
            print(net[i].get_shape().as_list())
            f.write(str(net[i].get_shape().as_list()) + "\n")
        f.close()

    def weighted_cross_entropy_c(self, label_c, prediction_c, class_weights):
        temp = -label_c * tf.math.log(prediction_c + 1e-3)#[Batch_size, patch_dimension, patc_dimension, 2]
        temp_weighted = class_weights * temp
        loss = tf.reduce_sum(temp_weighted, 3)
        return loss # [Batch_size, patch_dimension, patc_dimension, 1]

    def Learning_rate_decay(self):
        lr = self.args.lr / (1. + 10 * self.p)**0.75 #modificado de **0.75 para **0.95 - maior decaimento
        return lr

    def Train(self):
        # Best metric values to store
        best_val_fs = 0
        best_val_dr = 0
        best_val_dr_acc = 1
        best_mod_fs = 0
        best_mod_dr = 0
        #best_f1score = 0
        pat = 0
        
        # Default class weights to use in loss
        class_weights = []
        class_weights.append(0.4)
        class_weights.append(2)

        # Reference lists of source
        reference_t1_s = []
        reference_t2_s = []

        # Initialize lists of references of source with zero arrays on reference shape
        for s in self.dataset_s:
            reference_t1_s.append(np.zeros((s.references_[0].shape[0], s.references_[0].shape[1], 1)))
            reference_t2_s.append(np.zeros((s.references_[0].shape[0], s.references_[0].shape[1], 1)))
        
        # Reference lists of target
        reference_t1_t = []
        reference_t2_t = []

        # Initialize lists of references of target with zero arrays on reference shape
        for t in self.dataset_t:
            reference_t1_t.append(np.zeros((t.references_[0].shape[0], t.references_[0].shape[1], 1)))
            reference_t2_t.append(np.zeros((t.references_[0].shape[0], t.references_[0].shape[1], 1)))

        # If parameter is True, uses the positive percentage calculated as loss weights
        if self.args.balanced_tr:
            class_weights = self.dataset_s.class_weights

        # Corner coordinates list from Source
        corners_coordinates_tr_s = []
        corners_coordinates_vl_s = []

        # Copy values from object
        for index, s in enumerate(self.dataset_s):
            # Copy source corners coordinates from object and append it to another list
            corners_coordinates_tr_s.append(s.corners_coordinates_tr.copy())
            corners_coordinates_vl_s.append(s.corners_coordinates_vl.copy())

            # First swap 0s and 1s in reference T1 
            reference_t1_ = s.references_[0].copy()
            reference_t1_[s.references_[0] == 0] = 1
            reference_t1_[s.references_[0] == 1] = 0

            # Then copy reference arrays to initialized lists
            reference_t1_s[index][:,:,0] = reference_t1_.copy()
            reference_t2_s[index][:,:,0] = s.references_[1].copy()

        # Corner coordinates list from Target
        corners_coordinates_tr_t = []
        corners_coordinates_vl_t = []

        # Execute if it is Domain Adaptation
        if self.args.training_type == TRAINING_TYPE_DOMAIN_ADAPTATION:
            for t in self.dataset_t:
                corners_coordinates_tr_t.append(t.corners_coordinates_tr.copy())
                corners_coordinates_vl_t.append(t.corners_coordinates_vl.copy())

            if 'CL' in self.args.da_type:
                print("CL mode domain adaptation - Target domain will provide labels for training")
                for i in range(len(self.dataset_t)):
                    reference_t1_ = self.dataset_t[i].references_[0].copy()
                    reference_t1_[self.dataset_t[i].references_[0] == 0] = 1
                    reference_t1_[self.dataset_t[i].references_[0] == 1] = 0
                    reference_t1_t[i][:,:,0] = reference_t1_.copy()
                    reference_t2_t[i][:,:,0] = self.dataset_t[i].references_[1].copy()

        print('Sets dimensions before data augmentation')
        print('Source dimensions: ')
        for i in range(len(self.dataset_s)):
            print(np.shape(corners_coordinates_tr_s[i]))
            print(np.shape(corners_coordinates_vl_s[i]))

        print('Target dimensions: ')
        if self.args.training_type == TRAINING_TYPE_DOMAIN_ADAPTATION:
            for i in range(len(self.dataset_t)):
                print(np.shape(corners_coordinates_tr_t[i]))
                print(np.shape(corners_coordinates_vl_t[i]))
            print('******************************')
        
        # Do data augmentation definition
        if self.args.data_augmentation:
            print('Sets dimensions after data augmentation')
            print('Source dimensions: ')

            for i in range(len(self.dataset_s)):
                corners_coordinates_tr_s[i] = Data_Augmentation_Definition(corners_coordinates_tr_s[i])
                corners_coordinates_vl_s[i] = Data_Augmentation_Definition(corners_coordinates_vl_s[i])
                print(np.shape(corners_coordinates_tr_s[i]))
                print(np.shape(corners_coordinates_vl_s[i]))

            # Do data augmentation definition in target domain if it is domain adaptation
            print('Target dimensions: ')
            if self.args.training_type == TRAINING_TYPE_DOMAIN_ADAPTATION:
                for i in range(len(self.dataset_t)):
                    corners_coordinates_tr_t[i] = Data_Augmentation_Definition(corners_coordinates_tr_t[i])
                    corners_coordinates_vl_t[i] = Data_Augmentation_Definition(corners_coordinates_vl_t[i])
                    print(np.shape(corners_coordinates_tr_t[i]))
                    print(np.shape(corners_coordinates_vl_t[i]))

        # If it is domain adaptation, generate target labels
        if self.args.training_type == TRAINING_TYPE_DOMAIN_ADAPTATION and 'DR' in self.args.da_type:
            #Generating target labels before data shuffling and balancing            
            # Target Domain labels configuration
            # Source: 0
            # Target 1: 1
            # Target 2: 2                
            target_labels_tr = []                
            target_labels_vl = []     

            source_labels_tr = []
            source_labels_vl = []


            domain_indexs_tr_s = []
            domain_indexs_vl_s = []

            source_label_value = 0
            for i in range(len(self.dataset_s)):
                print(f"Source Dataset {self.dataset_s[i].DATASET}")

                if self.args.discriminate_domain_targets:
                    print(f"Assigned label value: {source_label_value}")
                    if len(self.D_out_shape) > 2:
                        source_labels_tr.append(np.full((np.shape(corners_coordinates_tr_s[i])[0], self.D_out_shape[0], self.D_out_shape[1],1),source_label_value))
                        source_labels_vl.append(np.full((np.shape(corners_coordinates_vl_s[i])[0], self.D_out_shape[0], self.D_out_shape[1],1),source_label_value))
                    else:
                        source_labels_tr.append(np.full((corners_coordinates_tr_s[i].shape[0], 1), source_label_value))
                        source_labels_vl.append(np.full((corners_coordinates_vl_s[i].shape[0], 1), source_label_value))
                else:
                    print(f"Assigned label value: 0")
                    if len(self.D_out_shape) > 2:
                        source_labels_tr.append(np.zeros((np.shape(corners_coordinates_tr_s[i])[0], self.D_out_shape[0], self.D_out_shape[1],1)))
                        source_labels_vl.append(np.zeros((np.shape(corners_coordinates_vl_s[i])[0], self.D_out_shape[0], self.D_out_shape[1],1)))
                    else:
                        source_labels_tr.append(np.zeros((corners_coordinates_tr_s[i].shape[0], 1)))
                        source_labels_vl.append(np.zeros((corners_coordinates_vl_s[i].shape[0], 1)))

                print(f"Assigned domain index value: {source_label_value}")
                # Source Domain indexs configuration
                domain_indexs_tr_s.append(np.full((np.shape(corners_coordinates_tr_s[i])[0], 1), source_label_value))
                domain_indexs_vl_s.append(np.full((np.shape(corners_coordinates_vl_s[i])[0], 1), source_label_value))

                source_label_value += 1


            domain_indexs_tr_t = []
            domain_indexs_vl_t = []
            
            target_label_value = source_label_value
            for i in range(len(self.dataset_t)):
                print(f"Target Dataset {self.dataset_t[i].DATASET}")

                if self.args.discriminate_domain_targets:
                    print("Assigned label value: %d"%(target_label_value))
                    if len(self.D_out_shape) > 2:
                        target_labels_tr.append(np.full((corners_coordinates_tr_t[i].shape[0], self.D_out_shape[0], self.D_out_shape[1],1),target_label_value))
                        target_labels_vl.append(np.full((corners_coordinates_vl_t[i].shape[0], self.D_out_shape[0], self.D_out_shape[1],1),target_label_value))
                    else:
                        target_labels_tr.append(np.full((corners_coordinates_tr_t[i].shape[0], 1), target_label_value))
                        target_labels_vl.append(np.full((corners_coordinates_vl_t[i].shape[0], 1), target_label_value))
                else:
                    print(f"Assigned label value: 1")
                    if len(self.D_out_shape) > 2:
                        target_labels_tr.append(np.ones((corners_coordinates_tr_t[i].shape[0], self.D_out_shape[0], self.D_out_shape[1],1)))
                        target_labels_vl.append(np.ones((corners_coordinates_vl_t[i].shape[0], self.D_out_shape[0], self.D_out_shape[1],1)))
                    else:
                        target_labels_tr.append(np.ones((corners_coordinates_tr_t[i].shape[0], 1)))
                        target_labels_vl.append(np.ones((corners_coordinates_vl_t[i].shape[0], 1)))
                
                print(f"Assigned domain index value: {target_label_value}")
                # Target Domains indexes configuration                    
                domain_indexs_tr_t.append(np.full((corners_coordinates_tr_t[i].shape[0], 1), target_label_value))
                domain_indexs_vl_t.append(np.full((corners_coordinates_vl_t[i].shape[0], 1), target_label_value))
                
                target_label_value += 1

            size_tr_s = [x.shape[0] for x in corners_coordinates_tr_s]
            size_vl_s = [x.shape[0] for x in corners_coordinates_vl_s]

            size_tr_t = [x.shape[0] for x in corners_coordinates_tr_t]
            size_vl_t = [x.shape[0] for x in corners_coordinates_vl_t]

            size_tr_list = np.concatenate((size_tr_s,size_tr_t),axis=0)
            size_vl_list = np.concatenate((size_vl_s,size_vl_t),axis=0)

            index_min_size_tr = np.argmin(size_tr_list)
            index_min_size_vl = np.argmin(size_vl_list)

            min_tr_size = size_tr_list[index_min_size_tr]
            min_vl_size = size_vl_list[index_min_size_vl]  

            print('min size_tr_list:')
            print(min_tr_size)

            print('min size_vl_list:')
            print(min_vl_size)
           
            
            data = []

            #Shuffling the num_samples
            index_tr_s = []
            index_vl_s = []

            for i in range(len(self.dataset_s)):
                index_tr_s.append(np.arange(size_tr_s[i]))
                index_vl_s.append(np.arange(size_vl_s[i]))

                np.random.shuffle(index_tr_s[i])
                np.random.shuffle(index_vl_s[i])

                corners_coordinates_tr_s[i] = corners_coordinates_tr_s[i][index_tr_s[i], :]
                corners_coordinates_vl_s[i] = corners_coordinates_vl_s[i][index_vl_s[i], :]    

                source_labels_tr[i] = source_labels_tr[i][index_tr_s[i], :]
                source_labels_vl[i] = source_labels_vl[i][index_vl_s[i], :]

                domain_indexs_tr_s[i] = domain_indexs_tr_s[i][index_tr_s[i], :]
                domain_indexs_vl_s[i] = domain_indexs_vl_s[i][index_vl_s[i], :]

                #RETIRA AMOSTRAS DO CONJUNTO MAIOR PARA SE IGUALAR AO MENOR            
                corners_coordinates_tr_s[i] = corners_coordinates_tr_s[i][:min_tr_size,:]    
                corners_coordinates_vl_s[i] = corners_coordinates_vl_s[i][:min_vl_size,:]   

                source_labels_tr[i] = source_labels_tr[i][:min_tr_size, :]     
                source_labels_vl[i] = source_labels_vl[i][:min_vl_size,:]

                domain_indexs_tr_s[i] = domain_indexs_tr_s[i][:min_tr_size, :]
                domain_indexs_vl_s[i] = domain_indexs_vl_s[i][:min_vl_size,:]

                x_train_s = np.concatenate((self.dataset_s[i].images_norm_[0], self.dataset_s[i].images_norm_[1], reference_t1_s[i], reference_t2_s[i]), axis = 2)
                data.append(x_train_s)


            index_tr_t = []
            index_vl_t = []
            
            for i in range(len(self.dataset_t)):
                index_tr_t.append(np.arange(size_tr_t[i]))
                index_vl_t.append(np.arange(size_vl_t[i]))
                
                np.random.shuffle(index_tr_t[i])
                np.random.shuffle(index_vl_t[i])

                corners_coordinates_tr_t[i] = corners_coordinates_tr_t[i][index_tr_t[i], :]
                corners_coordinates_vl_t[i] = corners_coordinates_vl_t[i][index_vl_t[i], :]
            
                target_labels_tr[i] = target_labels_tr[i][index_tr_t[i], :]
                target_labels_vl[i] = target_labels_vl[i][index_vl_t[i], :]

                domain_indexs_tr_t[i] = domain_indexs_tr_t[i][index_tr_t[i], :]
                domain_indexs_vl_t[i] = domain_indexs_vl_t[i][index_vl_t[i], :]

                #RETIRA AMOSTRAS DO CONJUNTO MAIOR PARA SE IGUALAR AO MENOR      
                corners_coordinates_tr_t[i] = corners_coordinates_tr_t[i][:min_tr_size,:]
                target_labels_tr[i] = target_labels_tr[i][:min_tr_size, :]
                domain_indexs_tr_t[i] = domain_indexs_tr_t[i][:min_tr_size, :]
                    
                corners_coordinates_vl_t[i] = corners_coordinates_vl_t[i][:min_vl_size,:]
                target_labels_vl[i] = target_labels_vl[i][:min_vl_size,:]
                domain_indexs_vl_t[i] = domain_indexs_vl_t[i][:min_vl_size,:]    

                x_train_t = np.concatenate((self.dataset_t[i].images_norm_[0], self.dataset_t[i].images_norm_[1], reference_t1_t[i], reference_t2_t[i]), axis = 2)
                data.append(x_train_t)
                       
            corners_coordinates_tr_s = np.concatenate(corners_coordinates_tr_s, axis=0)
            corners_coordinates_vl_s = np.concatenate(corners_coordinates_vl_s, axis=0)
                    
            corners_coordinates_tr_t = np.concatenate(corners_coordinates_tr_t, axis=0)
            corners_coordinates_vl_t = np.concatenate(corners_coordinates_vl_t, axis=0)

            corners_coordinates_tr = np.concatenate((corners_coordinates_tr_s, corners_coordinates_tr_t), axis = 0)
            corners_coordinates_vl = np.concatenate((corners_coordinates_vl_s, corners_coordinates_vl_t), axis = 0)
            
            domain_indexs_tr_s = np.concatenate(domain_indexs_tr_s,axis=0)
            domain_indexs_vl_s = np.concatenate(domain_indexs_vl_s,axis=0)

            domain_indexs_tr_t = np.concatenate(domain_indexs_tr_t,axis=0)
            domain_indexs_vl_t = np.concatenate(domain_indexs_vl_t,axis=0)

            domain_indexs_tr = np.concatenate((domain_indexs_tr_s, domain_indexs_tr_t), axis = 0)
            domain_indexs_vl = np.concatenate((domain_indexs_vl_s, domain_indexs_vl_t), axis = 0)

            source_labels_tr = np.concatenate(source_labels_tr,axis=0)
            source_labels_vl = np.concatenate(source_labels_vl,axis=0)

            target_labels_tr = np.concatenate(target_labels_tr,axis=0)
            target_labels_vl = np.concatenate(target_labels_vl,axis=0)

                
            y_train_d = np.concatenate((source_labels_tr, target_labels_tr), axis = 0)
            y_valid_d = np.concatenate((source_labels_vl, target_labels_vl), axis = 0)
        
        # Training configuration for Classification
        elif self.args.training_type == TRAINING_TYPE_CLASSIFICATION:
            # Create data variable and fill it with all images and references concatenaned in an unique array
            data = []
            for i in range(len(self.dataset_s)):
                x_train_s = np.concatenate((self.dataset_s[i].images_norm_[0], self.dataset_s[i].images_norm_[1], reference_t1_s[i], reference_t2_s[i]), axis = 2)
                data.append(x_train_s)
            
            # Initialize corner coordinates lists
            corners_coordinates_tr = []
            corners_coordinates_vl = []

            # Initialize domain indexes lists
            domain_indexs_tr = []
            domain_indexs_vl = []

            # Fill Domain_indexes lists with column array of source_label_value 
            # with length of corner coordinates
            source_label_value = 0
            for i in range(len(self.dataset_s)):
                corners_coordinates_tr.append(corners_coordinates_tr_s[i].copy())
                corners_coordinates_vl.append(corners_coordinates_vl_s[i].copy())

                domain_indexs_tr.append(np.full((np.shape(corners_coordinates_tr[i])[0], 1), source_label_value))
                domain_indexs_vl.append(np.full((np.shape(corners_coordinates_vl[i])[0], 1), source_label_value))

                source_label_value += 1

            corners_coordinates_tr = np.concatenate(corners_coordinates_tr, axis = 0)
            corners_coordinates_vl = np.concatenate(corners_coordinates_vl, axis = 0)

            domain_indexs_tr = np.concatenate(domain_indexs_tr,axis=0)
            domain_indexs_vl = np.concatenate(domain_indexs_vl,axis=0)
            
        # Computing the number of batches
        num_batches_tr = corners_coordinates_tr.shape[0]//self.args.batch_size
        num_batches_vl = corners_coordinates_vl.shape[0]//self.args.batch_size      

        print(f'Num samples for training: {corners_coordinates_tr.shape[0]}')
        print(f'Num samples for validation: {corners_coordinates_vl.shape[0]}')   
            
        # Append (another time in classification case) array to data list 
        data.append(x_train_s) 

        # Start training
        e = 0 # Current epoch
        best_model_epoch = -1 # Best model epoch
        while (e < self.args.epochs):
            # Shuffle train data and labels
            num_samples = corners_coordinates_tr.shape[0] # Number of train patches
            index = np.arange(num_samples)
            seed_value = 42
            rng = np.random.default_rng(seed=seed_value)
            rng.shuffle(index)
            # np.random.shuffle(index)
            corners_coordinates_tr = corners_coordinates_tr[index, :]
            domain_indexs_tr = domain_indexs_tr[index, :]

            # Execute if it is domain adaptation
            if self.args.training_type == TRAINING_TYPE_DOMAIN_ADAPTATION:
                if 'DR' in self.args.da_type:
                    if len(self.D_out_shape) > 2:
                        y_train_d = y_train_d[index, :, :, :]
                    else:
                        y_train_d = y_train_d[index, :]

            # Shuffle validation data and labels
            num_samples = corners_coordinates_vl.shape[0] # Number of valid patches
            index = np.arange(num_samples)
            rng = np.random.default_rng(seed=seed_value)
            rng.shuffle(index)
            # np.random.shuffle(index)
            corners_coordinates_vl = corners_coordinates_vl[index, :]
            domain_indexs_vl = domain_indexs_vl[index, :]

            # Execute if it is Domain Adaptation
            if self.args.training_type == TRAINING_TYPE_DOMAIN_ADAPTATION:
                if 'DR' in self.args.da_type:
                    if len(self.D_out_shape) > 2:
                        y_valid_d = y_valid_d[index, :, :, :]
                    else:
                        y_valid_d = y_valid_d[index, :]

            # Save training history in a file
            with open(os.path.join(self.args.save_checkpoint_path, "Log.txt"), "a") as f:
                # Initializing loss and metrics
                loss_cl_tr = np.zeros((1 , 2))
                loss_cl_vl = np.zeros((1 , 2))
                loss_dr_tr = np.zeros((1 , 2))
                loss_dr_vl = np.zeros((1 , 2))

                accuracy_tr = 0
                f1_score_tr = 0
                recall_tr = 0
                precission_tr = 0

                accuracy_vl = 0
                f1_score_vl = 0
                recall_vl = 0
                precission_vl = 0

                print("----------------------------------------------------------")
                #Computing some parameters
                self.p = float(e) / self.args.epochs
                print("Percentage of epochs: " + str(self.p))

                if self.args.training_type == 'domain_adaptation':
                    warmup = 1
                    if e >= warmup:
                        self.l = (2. / (1. + np.exp(-self.args.gamma * self.p)) - 1)
                    else:
                        self.l = 0.
                    print("lambda_p: " + str(self.l))

                self.lr = self.Learning_rate_decay()
                print("Learning rate decay: " + str(self.lr))                

                batch_counter_cl = 0
                batchs = trange(num_batches_tr)
                for b in batchs:
                    corners_coordinates_tr_batch = corners_coordinates_tr[b * self.args.batch_size : (b + 1) * self.args.batch_size , :]
                    domain_index_batch = domain_indexs_tr[b * self.args.batch_size : (b + 1) * self.args.batch_size, :]


                    #Domain mask: Source samples = 1 and Target samples = 0
                    domain_mask_segmentation_tr = np.zeros((self.args.batch_size, self.args.patches_dimension, self.args.patches_dimension, self.args.num_classes),dtype=np.float32)
                    domain_mask_segmentation_tr[:,:,:,:] = np.where(domain_index_batch != 0, 0., 1.).astype(np.float32)[:, np.newaxis, np.newaxis, :]

                    if self.args.data_augmentation:
                        transformation_indexs_batch = corners_coordinates_tr[b * self.args.batch_size : (b + 1) * self.args.batch_size , 4]

                    #Extracting the data patches from it's coordinates
                    data_batch_ = Patch_Extraction(data, corners_coordinates_tr_batch, domain_index_batch, self.args.patches_dimension)
                    # data_total_tr = Patch_Extraction(data, corners_coordinates_tr, domain_indexs_tr, self.args.patches_dimension)
                    
                    # Perform data augmentation?
                    if self.args.data_augmentation:
                        data_batch_ = Data_Augmentation_Execution(data_batch_, transformation_indexs_batch)
                        
                    # Recovering data
                    data_batch = data_batch_[:,:,:,: 2 * self.args.image_channels]
                    # x_data_tr = data_total_tr[:,:,:,: 2 * self.args.image_channels]
                    
                    # Recovering past reference
                    reference_t1_ = data_batch_[:,:,:, 2 * self.args.image_channels]
                    reference_t2_ = data_batch_[:,:,:, 2 * self.args.image_channels + 1]
                    # y_t1_tr = data_total_tr[:,:,:, 2 * self.args.image_channels]
                    # y_t2_tr = data_total_tr[:,:,:, 2 * self.args.image_channels + 1]
                    # y_t2_tr[y_t1_tr == 0] = 255 
                    
                    # Hot encoding the reference_t2_
                    y_train_c_hot_batch = tf.keras.utils.to_categorical(reference_t2_, self.args.num_classes)
                    classification_mask_batch = reference_t1_.copy()

                    # Setting the class weights
                    Weights = np.ones((self.args.batch_size, self.args.patches_dimension, self.args.patches_dimension, self.args.num_classes),dtype=np.float32)
                    Weights[:,:,:,0] = class_weights[0] * Weights[:,:,:,0]
                    Weights[:,:,:,1] = class_weights[1] * Weights[:,:,:,1]

                    if self.args.training_type == TRAINING_TYPE_CLASSIFICATION:
                        _, c_batch_loss, batch_probs = self.sess.run([self.training_optimizer, self.total_loss, self.prediction_c],
                                                                feed_dict={self.data: data_batch, self.label_c: y_train_c_hot_batch,
                                                                           self.mask_c: classification_mask_batch, self.class_weights: Weights, self.learning_rate: self.lr})
                    elif self.args.training_type == TRAINING_TYPE_DOMAIN_ADAPTATION:
                        if 'DR' in self.args.da_type:
                            if len(self.D_out_shape) > 2:
                                y_train_d_batch = y_train_d[b * self.args.batch_size : (b + 1) * self.args.batch_size, :, :,:]
                            else:
                                y_train_d_batch = y_train_d[b * self.args.batch_size : (b + 1) * self.args.batch_size, :]

                            y_train_d_hot_batch = tf.keras.utils.to_categorical(y_train_d_batch, self.num_domains)

                            _, c_batch_loss, batch_probs, d_batch_loss  = self.sess.run([self.training_optimizer, self.classifier_loss, self.prediction_c, self.domainregressor_loss],
                                                                        feed_dict={self.data: data_batch, self.label_c: y_train_c_hot_batch, self.label_d: y_train_d_hot_batch,
                                                                                    self.mask_c: classification_mask_batch, self.class_weights: Weights, self.L: self.l, self.learning_rate: self.lr})

                            loss_dr_tr[0 , 0] += d_batch_loss
                        else:
                            _, c_batch_loss, batch_probs  = self.sess.run([self.training_optimizer, self.total_loss, self.prediction_c],
                                                                        feed_dict={self.data: data_batch, self.label_c: y_train_c_hot_batch,
                                                                                    self.mask_c: classification_mask_batch, self.class_weights: Weights, self.learning_rate: self.lr})

                    
                    loss_cl_tr[0 , 0] += c_batch_loss
                    # print(loss_cl_tr)
                    y_train_predict_batch = np.argmax(batch_probs, axis = 3)
                    y_train_batch = np.argmax(y_train_c_hot_batch, axis = 3)

                    # Reshaping probability output, true labels and last reference
                    y_train_predict_r = y_train_predict_batch.reshape((y_train_predict_batch.shape[0] * y_train_predict_batch.shape[1] * y_train_predict_batch.shape[2], 1))
                    y_train_true_r = y_train_batch.reshape((y_train_batch.shape[0] * y_train_batch.shape[1] * y_train_batch.shape[2], 1))
                    classification_mask_batch_r = classification_mask_batch.reshape((classification_mask_batch.shape[0] * classification_mask_batch.shape[1] * classification_mask_batch.shape[2], 1))

                    available_training_pixels= np.transpose(np.array(np.where(classification_mask_batch_r == 1)))

                    y_predict = y_train_predict_r[available_training_pixels[:,0],available_training_pixels[:,1]]
                    y_true = y_train_true_r[available_training_pixels[:,0],available_training_pixels[:,1]]

                    accuracy, f1score, recall, precission, conf_mat = compute_metrics(y_true.astype(int), y_predict.astype(int))

                    accuracy_tr += accuracy
                    f1_score_tr += f1score
                    recall_tr += recall
                    precission_tr += precission

                    batch_counter_cl += 1

                loss_cl_tr = loss_cl_tr/batch_counter_cl
                accuracy_tr = accuracy_tr/batch_counter_cl
                f1_score_tr = f1_score_tr/batch_counter_cl
                recall_tr = recall_tr/batch_counter_cl
                precission_tr = precission_tr/batch_counter_cl
                loss_dr_tr = loss_dr_tr/batch_counter_cl

                self.segmentation_history["loss"].append(loss_cl_tr[0 , 0])       
                self.segmentation_history["f1"].append(f1_score_tr)

                print("batch_counter_cl:")
                print(batch_counter_cl)

                if self.args.training_type == TRAINING_TYPE_DOMAIN_ADAPTATION and 'DR' in self.args.da_type:                    
                    
                    self.discriminator_history["loss"].append(loss_dr_tr[0 , 0])
                    #self.discriminator_history["accuracy"].append(acc_discriminator_train)

                    print ("%d [Training loss: %f, acc.: %.2f%%, precision: %.2f%%, recall: %.2f%%, f1: %.2f%%, Dr loss: %f]" % (e, loss_cl_tr[0,0], accuracy_tr, precission_tr, recall_tr, f1_score_tr, loss_dr_tr[0,0]))
                    f.write("%d [Training loss: %f, acc.: %.2f%%, precision: %.2f%%, recall: %.2f%%, f1: %.2f%%, Dr loss: %f]\n" % (e, loss_cl_tr[0,0], accuracy_tr, precission_tr, recall_tr, f1_score_tr, loss_dr_tr[0,0]))
                else:
                    print ("%d [Training loss: %f, acc.: %.2f%%, precision: %.2f%%, recall: %.2f%%, f1: %.2f%%]" % (e, loss_cl_tr[0,0], accuracy_tr, precission_tr, recall_tr, f1_score_tr))
                    f.write("%d [Training loss: %f, acc.: %.2f%%, precision: %.2f%%, recall: %.2f%%, f1: %.2f%%]\n" % (e, loss_cl_tr[0,0], accuracy_tr, precission_tr, recall_tr, f1_score_tr))

                #Computing the validation loss
                print('[*]Computing the validation loss...')
                batch_counter_cl = 0
                batchs = trange(num_batches_vl)
                #for b in range(num_batches_vl):
                for b in batchs:
                    corners_coordinates_vl_batch = corners_coordinates_vl[b * self.args.batch_size : (b + 1) * self.args.batch_size , :]
                    domain_index_batch = domain_indexs_vl[b * self.args.batch_size : (b + 1) * self.args.batch_size, :]

                    if self.args.data_augmentation:
                        transformation_indexs_batch = corners_coordinates_vl[b * self.args.batch_size : (b + 1) * self.args.batch_size , 4]

                    #Extracting the data patches from it's coordinates
                    data_batch_ = Patch_Extraction(data, corners_coordinates_vl_batch, domain_index_batch, self.args.patches_dimension)
                    # data_total_vl = Patch_Extraction(data, corners_coordinates_vl, domain_indexs_vl, self.args.patches_dimension)
                    
                    if self.args.data_augmentation:
                        data_batch_ = Data_Augmentation_Execution(data_batch_, transformation_indexs_batch)

                    # Recovering data
                    data_batch = data_batch_[:,:,:,: 2 * self.args.image_channels]
                    # x_data_vl = data_total_vl[:,:,:,: 2 * self.args.image_channels]
                    
                    # Recovering past reference
                    reference_t1_ = data_batch_[:,:,:, 2 * self.args.image_channels]
                    reference_t2_ = data_batch_[:,:,:, 2 * self.args.image_channels + 1]
                    # y_t1_vl = data_total_vl[:,:,:, 2 * self.args.image_channels]
                    # y_t2_vl = data_total_vl[:,:,:, 2 * self.args.image_channels + 1] 
                    # y_t2_vl[y_t1_vl == 0] = 255 

                    # Hot encoding the reference_t2_
                    y_valid_c_hot_batch = tf.keras.utils.to_categorical(reference_t2_, self.args.num_classes)
                    classification_mask_batch = reference_t1_.copy()

                    # Setting the class weights
                    Weights = np.ones((corners_coordinates_vl_batch.shape[0], self.args.patches_dimension, self.args.patches_dimension, self.args.num_classes),dtype=np.float32)
                    Weights[:,:,:,0] = class_weights[0] * Weights[:,:,:,0]
                    Weights[:,:,:,1] = class_weights[1] * Weights[:,:,:,1]

                    #Domain mask: Source samples = 1 and Target samples = 0
                    domain_mask_segmentation_vl = np.zeros((self.args.batch_size, self.args.patches_dimension, self.args.patches_dimension, self.args.num_classes),dtype=np.float32)
                    domain_mask_segmentation_vl[:,:,:,:] = np.where(domain_index_batch != 0, 0., 1.).astype(np.float32)[:, np.newaxis, np.newaxis, :]

                    
                    if self.args.training_type == TRAINING_TYPE_CLASSIFICATION:
                        c_batch_loss, batch_probs = self.sess.run([self.total_loss, self.prediction_c],
                                                              feed_dict={self.data: data_batch, self.label_c: y_valid_c_hot_batch,
                                                                         self.mask_c: classification_mask_batch, self.class_weights: Weights,  self.learning_rate: self.lr})
                    if self.args.training_type == TRAINING_TYPE_DOMAIN_ADAPTATION:
                        if 'DR' in self.args.da_type:
                            if len(self.D_out_shape) > 2:
                                y_valid_d_batch = y_valid_d[b * self.args.batch_size : (b + 1) * self.args.batch_size, :, :,:]
                            else:
                                y_valid_d_batch = y_valid_d[b * self.args.batch_size : (b + 1) * self.args.batch_size, :]

                        #y_valid_d_batch = y_valid_d[b * self.args.batch_size : (b + 1) * self.args.batch_size, :]
                        y_valid_d_hot_batch = tf.keras.utils.to_categorical(y_valid_d_batch, self.num_domains)
                        c_batch_loss, batch_probs, d_batch_loss = self.sess.run([self.classifier_loss, self.prediction_c, self.domainregressor_loss],
                                                                                feed_dict={self.data: data_batch, self.label_c: y_valid_c_hot_batch, self.label_d: y_valid_d_hot_batch,
                                                                                self.mask_c: classification_mask_batch, self.class_weights: Weights, self.L: 0, self.learning_rate: self.lr})

                        loss_dr_vl[0 , 0] += d_batch_loss
                    else:
                        c_batch_loss, batch_probs = self.sess.run([self.total_loss, self.prediction_c],
                                                                  feed_dict={self.data: data_batch, self.label_c: y_valid_c_hot_batch,
                                                                             self.mask_c: classification_mask_batch, self.class_weights: Weights,  self.learning_rate: self.lr})                            

                    loss_cl_vl[0 , 0] += c_batch_loss

                    y_valid_batch = np.argmax(y_valid_c_hot_batch, axis = 3)
                    y_valid_predict_batch = np.argmax(batch_probs, axis = 3)

                    # Reshaping probability output, true labels and last reference
                    y_valid_predict_r = y_valid_predict_batch.reshape((y_valid_predict_batch.shape[0] * y_valid_predict_batch.shape[1] * y_valid_predict_batch.shape[2], 1))
                    y_valid_true_r = y_valid_batch.reshape((y_valid_batch.shape[0] * y_valid_batch.shape[1] * y_valid_batch.shape[2], 1))
                    classification_mask_batch_r = classification_mask_batch.reshape((classification_mask_batch.shape[0] * classification_mask_batch.shape[1] * classification_mask_batch.shape[2], 1))

                    available_validation_pixels = np.transpose(np.array(np.where(classification_mask_batch_r == 1)))

                    y_predict = y_valid_predict_r[available_validation_pixels[:,0],available_validation_pixels[:,1]]
                    y_true = y_valid_true_r[available_validation_pixels[:,0],available_validation_pixels[:,1]]

                    accuracy, f1score, recall, precission, conf_mat = compute_metrics(y_true.astype(int), y_predict.astype(int))

                    accuracy_vl += accuracy
                    f1_score_vl += f1score
                    recall_vl += recall
                    precission_vl += precission
                    batch_counter_cl += 1

                loss_cl_vl = loss_cl_vl/(batch_counter_cl)
                accuracy_vl = accuracy_vl/(batch_counter_cl)
                f1_score_vl = f1_score_vl/(batch_counter_cl)
                recall_vl = recall_vl/(batch_counter_cl)
                precission_vl = precission_vl/(batch_counter_cl)
                loss_dr_vl = loss_dr_vl/batch_counter_cl

                self.segmentation_history["val_loss"].append(loss_cl_vl[0 , 0])      
                self.segmentation_history["val_f1"].append(f1_score_vl)

                if self.args.training_type == TRAINING_TYPE_DOMAIN_ADAPTATION and 'DR' in self.args.da_type:                    
                    
                    self.discriminator_history["val_loss"].append(loss_dr_vl[0 , 0])
                    #self.discriminator_history["val_accuracy"].append(acc_discriminator_val)

                    print ("%d [Validation loss: %f, acc.: %.2f%%,  precision: %.2f%%, recall: %.2f%%, f1: %.2f%%, DrV loss: %f]" % (e, loss_cl_vl[0,0], accuracy_vl, precission_vl, recall_vl, f1_score_vl, loss_dr_vl[0,0]))
                    f.write("%d [Validation loss: %f, acc.: %.2f%%, precision: %.2f%%, recall: %.2f%%, f1: %.2f%%, DrV loss: %f]\n" % (e, loss_cl_vl[0,0], accuracy_vl, precission_vl, recall_vl, f1_score_vl, loss_dr_vl[0,0]))
                else:
                    print ("%d [Validation loss: %f, acc.: %.2f%%,  precision: %.2f%%, recall: %.2f%%, f1: %.2f%%]" % (e, loss_cl_vl[0,0], accuracy_vl, precission_vl, recall_vl, f1_score_vl))
                    f.write("%d [Validation loss: %f, acc.: %.2f%%, precision: %.2f%%, recall: %.2f%%, f1: %.2f%%]\n" % (e, loss_cl_vl[0,0], accuracy_vl, precission_vl, recall_vl, f1_score_vl))

            if self.args.training_type == TRAINING_TYPE_DOMAIN_ADAPTATION:
                with open(os.path.join(self.args.save_checkpoint_path,"Log.txt"),"a") as f:
                    if 'DR' in self.args.da_type:
                        if np.isnan(loss_cl_tr[0,0]) or np.isnan(loss_cl_vl[0,0]) or np.isnan(loss_dr_tr[0,0]) or np.isnan(loss_dr_vl[0,0]):
                            print('Nan value detected!!!!')                            
                            if best_model_epoch != -1:
                                pat += 1
                                if pat > self.args.patience:
                                    print("Patience limit reached. Exiting training...")
                                    break
                                
                                print('[*]ReLoading the models weights...')
                                
                                self.sess.run(tf.compat.v1.initialize_all_variables())
                                mod = self.load(self.args.save_checkpoint_path)
                                if mod:
                                    print(" [*] Load with SUCCESS")
                                else:
                                    print(" [!] Load failed...")
                                    self.sess.run(tf.compat.v1.initialize_all_variables())
                            else:
                                print('There is no model to be loaded. Reinitializing variables...')
                                self.sess.run(tf.compat.v1.initialize_all_variables())
                        elif self.l != 0:
                            FLAG = False
                            if  best_val_dr < loss_dr_vl[0 , 0] and loss_dr_vl[0 , 0] < self.loss_dr_threshold:
                                if best_val_fs < f1_score_vl:
                                    best_val_dr = loss_dr_vl[0 , 0]
                                    best_val_fs = f1_score_vl
                                    best_mod_fs = f1_score_vl
                                    best_mod_dr = loss_dr_vl[0 , 0]
                                    best_model_epoch = e
                                    print('[!]Saving best ideal model at epoch: ' + str(e))
                                    f.write("[!]Ideal best ideal model\n")                                    
                                    self.save(self.args.save_checkpoint_path, best_model_epoch)
                                    FLAG = True
                                elif np.abs(best_val_fs - f1_score_vl) < 3:
                                    best_val_dr = loss_dr_vl[0 , 0]
                                    best_mod_fs = f1_score_vl
                                    best_mod_dr = loss_dr_vl[0 , 0]
                                    best_model_epoch = e
                                    print('[!]Saving best model attending best Dr_loss at epoch: ' + str(e))
                                    f.write("[!]Best model attending best Dr_loss\n")                                    
                                    self.save(self.args.save_checkpoint_path, best_model_epoch)
                                    FLAG = True
                            elif best_val_fs < f1_score_vl:
                                if  np.abs(best_val_dr - loss_dr_vl[0 , 0]) < 0.2:
                                    best_val_fs = f1_score_vl
                                    best_mod_fs = f1_score_vl
                                    best_mod_dr = loss_dr_vl[0 , 0]
                                    best_model_epoch = e
                                    print('[!]Saving best model attending best f1-score at epoch: ' + str(e))
                                    f.write("[!]Best model attending best f1-score \n")
                                    self.save(self.args.save_checkpoint_path, best_model_epoch)
                                    FLAG = True

                            if FLAG:
                                pat = 0
                                print('[!] Best Model with DrV loss: %.3f and F1-Score: %.2f%%'% (best_mod_dr, best_mod_fs))
                            else:
                                print('[!] The Model has not been considered as suitable for saving procedure.')
                                if best_model_epoch != -1:
                                    pat += 1
                                    if pat > self.args.patience:
                                        print("Patience limit reached. Exiting training...")
                                        break
                        else:
                            print("Warming up!")
                    else:
                        if best_val_fs < f1_score_vl:
                            best_val_fs = f1_score_vl
                            pat = 0
                            print('[!] Best Validation F1 score: %.2f%%'%(best_val_fs))
                            best_model_epoch = e
                            if self.args.save_intermediate_model:
                                print('[!]Saving best model at epoch: ' + str(e))                                
                                self.save(self.args.save_checkpoint_path, best_model_epoch)
                        else:
                            pat += 1
                            if pat > self.args.patience:
                                print("Patience limit reached. Exiting training...")
                                break
            else:
                if best_val_fs < f1_score_vl:
                    best_val_fs = f1_score_vl
                    pat = 0
                    print('[!] Best Validation F1 score: %.2f%%'%(best_val_fs))
                    best_model_epoch = e
                    if self.args.save_intermediate_model:
                        print('[!]Saving best model at epoch: ' + str(e))                        
                        self.save(self.args.save_checkpoint_path, best_model_epoch)
                else:
                    pat += 1
                    if pat > self.args.patience:
                        print("Patience limit reachead. Exiting training...")
                        break             
            e += 1

        
        if self.args.training_type == TRAINING_TYPE_CLASSIFICATION:
            self.plot_metrics_segmentation()
            with open(os.path.join(self.args.save_checkpoint_path,"Log.txt"),"a") as f:
                print('Training ended')
                f.write("Training ended\n")
                print("[!] Best Validation F1 score: %.2f%%"%(best_val_fs))
                f.write("[!] Best Validation F1 score: %.2f%%"%(best_val_fs))

        elif self.args.training_type == TRAINING_TYPE_DOMAIN_ADAPTATION:
            self.plot_metrics_segmentation()
            with open(os.path.join(self.args.save_checkpoint_path,"Log.txt"),"a") as f:
                if best_model_epoch != -1:
                    print("Training ended")
                    print("[!] Best epoch: %d" %(best_model_epoch))
                    print("[!] Domain Regressor Validation F1-score: %.2f%%" % (best_mod_fs))
                    print("[!] DrV loss: %.3f" % (best_val_dr))
                    print("[!] Best DR Validation for higher DrV loss: %.3f and F1-Score: %.2f%%" % (best_mod_dr, best_mod_fs))

                    f.write("Training ended\n")
                    f.write("[!] Best epoch: %d\n" %(best_model_epoch))
                    f.write("[!] Domain Regressor Validation F1-score: %.2f%%: \n" % (best_mod_fs))
                    f.write("[!] DrV loss: %.3f: \n" % (best_val_dr))
                else:
                    print("Training ended")
                    print("[!] [!] No Model has been selected.")
                    print("loss_dr_vl: %.3f"%(loss_dr_vl[0 , 0]))                    
                    print("f1_score_vl: %.3f"%(f1_score_vl))                    

                    f.write("Training ended")
                    f.write("[!] [!] No Model has been selected.")
                    f.write("loss_dr_vl: %.3f \n"%(loss_dr_vl[0 , 0]))                    
                    f.write("f1_score_vl: %.3f \n"%(f1_score_vl))
        

    def Test(self):

        for ds in self.dataset_t:

            hit_map_ = np.zeros((ds.k1 * ds.stride, ds.k2 * ds.stride))

            x_test = []
            data = np.concatenate((ds.images_norm_[0], ds.images_norm_[1]), axis = 2)
            x_test.append(data)

            num_batches_ts = ds.corners_coordinates_ts.shape[0]//self.args.batch_size
            batchs = trange(num_batches_ts)
            print(num_batches_ts)

            for b in batchs:
                self.corners_coordinates_ts_batch = ds.corners_coordinates_ts[b * self.args.batch_size : (b + 1) * self.args.batch_size , :]
                #self.x_test_batch = Patch_Extraction(x_test, self.central_pixels_coor_ts_batch, np.zeros((self.args.batch_size , 1)), self.args.patches_dimension, True, 'reflect')
                self.x_test_batch = Patch_Extraction(x_test, self.corners_coordinates_ts_batch, np.zeros((self.args.batch_size , 1)), self.args.patches_dimension)
                
                probs = self.sess.run(self.prediction_c,feed_dict={self.data: self.x_test_batch})

                for i in range(self.args.batch_size):
                    hit_map_[int(self.corners_coordinates_ts_batch[i, 0]) : int(self.corners_coordinates_ts_batch[i, 0]) + int(ds.stride),
                        int(self.corners_coordinates_ts_batch[i, 1]) : int(self.corners_coordinates_ts_batch[i, 1]) + int(ds.stride)] = probs[i, int(ds.overlap//2) : int(ds.overlap//2) + int(ds.stride),
                                                                                                                                                           int(ds.overlap//2) : int(ds.overlap//2) + int(ds.stride),1]

            if (num_batches_ts * self.args.batch_size) < ds.corners_coordinates_ts.shape[0]:
                self.corners_coordinates_ts_batch = ds.corners_coordinates_ts[num_batches_ts * self.args.batch_size : , :]
                self.x_test_batch = Patch_Extraction(x_test, self.corners_coordinates_ts_batch, np.zeros((self.corners_coordinates_ts_batch.shape[0] , 1)), self.args.patches_dimension)

                probs = self.sess.run(self.prediction_c,feed_dict={self.data: self.x_test_batch})

                for i in range(self.corners_coordinates_ts_batch.shape[0]):
                    hit_map_[int(self.corners_coordinates_ts_batch[i, 0]) : int(self.corners_coordinates_ts_batch[i, 0]) + int(ds.stride),
                        int(self.corners_coordinates_ts_batch[i, 1]) : int(self.corners_coordinates_ts_batch[i, 1]) + int(ds.stride)] = probs[i, int(ds.overlap//2) : int(ds.overlap//2) + int(ds.stride),
                                                                                                                                                           int(ds.overlap//2) : int(ds.overlap//2) + int(ds.stride),1]
            hit_map = hit_map_[:ds.k1 * ds.stride - ds.step_row, :ds.k2 * ds.stride - ds.step_col]
            
            print("Hit map:")
            print(np.shape(hit_map))
            np.save(os.path.join(self.args.save_results_dir,'hit_map'), hit_map)

    def save(self, checkpoint_dir, epoch):

        # TODO: Implement if else for saving DeepLab or Unet
        if self.args.classifier_type == 'Unet': #if/elif inserido
            model_name = "Unet"
        elif self.args.classifier_type == 'SegNet': #if/elif inserido
            model_name = "SegNet"
        elif self.args.classifier_type == 'DeepLab': #if/elif inserido
            model_name = "DeepLab"
        # Not saving because of google colab
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=epoch)
        print("Checkpoint Saved with SUCCESS!")

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")
        print(checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            aux = 'model_example'
            for i in range(len(ckpt_name)):
                if ckpt_name[-i-1] == '-':
                    aux = ckpt_name[-i:]
                    break
            return aux
        else:
            return ''

    def plot_metrics_segmentation(self):
        plt.figure(figsize=(10, 10))
        plt.plot(self.segmentation_history["loss"], label="decoder training loss")        
        plt.plot(self.segmentation_history["val_loss"], label="decoder validation loss")  
        plt.plot(self.discriminator_history["loss"], label="discriminator training loss")    
        plt.plot(self.discriminator_history["val_loss"], label="discriminator validation loss")           
        plt.title("Loss evolution")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.ylim(0, 2)
        plt.legend()
        plt.savefig(os.path.join(self.args.save_checkpoint_path,"..","segmentation_metrics_run_"+self.args.num_run+".png"))

    def plot_f1score_segmentation(self):
        plt.figure(figsize=(10, 10))        
        plt.plot(self.segmentation_history["f1"], label="training f1score")        
        plt.plot(self.segmentation_history["val_f1"], label="validation f1score")
        plt.plot(self.discriminator_history["accuracy"], label="discriminator training accuracy")    
        plt.plot(self.discriminator_history["val_accuracy"], label="discriminator validation accuracy") 

        plt.title("Segmentation F1 vs Discriminator Acc evolution")
        plt.xlabel("Epoch #")
        plt.ylabel("F1")
        plt.ylim([0, 100])
        plt.legend()
        plt.savefig(os.path.join(self.args.save_checkpoint_path,"segmentation_f1score.png"))
    

def Metrics_For_Test(hit_map, uncertainty_map,
                     reference_t1, reference_t2,
                     Train_tiles, Valid_tiles, Undesired_tiles,
                     Thresholds, image_t2,
                     args):

    #audit_area = [i for i in range(10)]
    
    audit_area = 3.0

    save_path = os.path.join(args.results_dir, args.file)
    print('[*]Defining the initial central patches coordinates...')
    
    print('Undesired_tiles: ')
    print(Undesired_tiles)
    #mask_init = mask_creation(reference_t1.shape[0], reference_t1.shape[1], args.horizontal_blocks, args.vertical_blocks, [], [], [])
    mask_original = mask_creation(reference_t1.shape[0], reference_t1.shape[1], args.horizontal_blocks, args.vertical_blocks, Train_tiles, Valid_tiles, Undesired_tiles)

    mask_final = mask_original.copy()
    
    mask_final[mask_final == 1] = 0
    mask_final[mask_final == 3] = 0
    mask_final[mask_final == 2] = 1
    
    np.save(os.path.join(save_path,f'mask_final_{args.dataset}.npy'), mask_final)
    
    Probs_init = hit_map

    if Thresholds is None:
        reference_t1_copy_ = reference_t1.copy()
        reference_t1_copy_ = reference_t1_copy_ - 1
        reference_t1_copy_[reference_t1_copy_ == -1] = 1
        reference_t1_copy_[reference_t2 == 2] = 0
        mask_f_ = mask_final * reference_t1_copy_
        # Raul Implementation
        min_array = np.zeros((1 , ))
        Pmax = np.max(Probs_init[mask_f_ == 1])
        probs_list = np.arange(Pmax, 0, -Pmax/(args.Npoints - 1))
        Thresholds = np.concatenate((probs_list , min_array))

    # Metrics containers
    ACCURACY = np.zeros((1, len(Thresholds)))
    FSCORE = np.zeros((1, len(Thresholds)))
    RECALL = np.zeros((1, len(Thresholds)))
    PRECISSION = np.zeros((1, len(Thresholds)))
    CONFUSION_MATRIX = np.zeros((2 , 2, len(Thresholds)))
    ALERT_AREA = np.zeros((1 , len(Thresholds)))
    UNCERTAINTY_MEAN = np.zeros((1, len(Thresholds)))
    FSCORE_LOW_UNCERTAINTY = np.zeros((1, len(Thresholds)))
    FSCORE_HIGH_UNCERTAINTY = np.zeros((1, len(Thresholds)))
    FSCORE_AUDIT = np.zeros((1, len(Thresholds)))
    AUDIT_THRESHOLD = np.zeros((1, len(Thresholds)))

    print('[*]The metrics computation has started...')
    #Computing the metrics for each defined threshold
    for th in range(len(Thresholds)):
        print(Thresholds[th])

        positive_map_init = np.zeros_like(hit_map)
        reference_t1_copy = reference_t1.copy()
        
        #Adding previously deforestated area to the mask
        mask_original[reference_t1_copy == 1] = 4

        threshold = Thresholds[th]
        positive_coordinates = np.transpose(np.array(np.where(Probs_init >= threshold)))
        positive_map_init[positive_coordinates[:,0].astype('int'), positive_coordinates[:,1].astype('int')] = 1

        if args.eliminate_regions:
            positive_map_init_ = skimage.morphology.area_opening(positive_map_init.astype('int'),area_threshold = args.area_avoided, connectivity=1)
            eliminated_samples = positive_map_init - positive_map_init_
        else:
            eliminated_samples = np.zeros_like(hit_map)


        reference_t1_copy = reference_t1_copy + eliminated_samples
        reference_t1_copy[reference_t1_copy == 2] = 1

        reference_t1_copy = reference_t1_copy - 1
        reference_t1_copy[reference_t1_copy == -1] = 1
        reference_t1_copy[reference_t2 == 2] = 0
        mask_f = mask_final * reference_t1_copy
        
        print('mask_f shape:')
        print(np.shape(mask_f))
        
        brown_colour = [204, 102, 0]

        #central_pixels_coordinates_ts, y_test = Central_Pixel_Definition_For_Test(mask_final_, reference_t1_copy, reference_t2, args.patches_dimension, 1, 'metrics')
        central_pixels_coordinates_ts_ = np.transpose(np.array(np.where(mask_f == 1)))
        
        y_test = reference_t2[central_pixels_coordinates_ts_[:,0].astype('int'), central_pixels_coordinates_ts_[:,1].astype('int')]

        #print(np.shape(central_pixels_coordinates_ts))
        Probs = hit_map[central_pixels_coordinates_ts_[:,0].astype('int'), central_pixels_coordinates_ts_[:,1].astype('int')]
        
        Probs[Probs >= Thresholds[th]] = 1
        Probs[Probs <  Thresholds[th]] = 0
        
        if uncertainty_map is not None:
            uncertainty_map = np.clip(uncertainty_map, 0, 1)
            
            uncertaint_map_ = uncertainty_map[central_pixels_coordinates_ts_[:,0].astype('int'), central_pixels_coordinates_ts_[:,1].astype('int')]
            UNCERTAINTY_MEAN[0 , th] = np.mean(uncertaint_map_.flatten())*100
            
            uncertaint_map_ = np.maximum(uncertaint_map_, 0)
            
            uncertainty_threshold = np.percentile(uncertaint_map_, 100.0 - audit_area)
            mask_uncertainty = uncertaint_map_ >= uncertainty_threshold
            high_uncertainty_mask = np.where(mask_uncertainty, 1, 0)
            
            _, f1score_low, f1score_high, f1score_audit = compute_f1_uncertainty(y_test.astype('int'), Probs.astype('int'), high_uncertainty_mask)
            
            audit_predictions = compute_audit_mask(y_test.astype('int'), Probs.astype('int'), high_uncertainty_mask)
            
            if args.create_classification_map and len(Thresholds) == 1 and image_t2 is not None:
                selected_channels = image_t2[:, :, [3, 2, 1]]
                selected_channels[:,:,0] *= mask_f
                selected_channels[:,:,1] *= mask_f
                selected_channels[:,:,2] *= mask_f
                np.save(os.path.join(save_path,'image_t2_mask'), selected_channels)
                Classification_map_audit, _, _ = Classification_Maps(audit_predictions.astype('int'), y_test.astype('int'), central_pixels_coordinates_ts_, hit_map)
                
                # Locate where the mask equals 4
                Classification_map_audit[mask_original == 4] = brown_colour
                
                plt.imsave(os.path.join(save_path,'Classification_map_audit.png'), Classification_map_audit)
                
                cmap = plt.cm.get_cmap('jet')
                # Convert normalized data to RGB using the colormap
                rgb_image = cmap(uncertainty_map)
                # Remove the alpha channel (the fourth dimension)
                rgb_image = rgb_image[..., :3]
                plt.imsave(os.path.join(save_path,' .png'), uncertainty_map, cmap='jet')
            
            FSCORE_LOW_UNCERTAINTY[0 , th] = f1score_low
            FSCORE_HIGH_UNCERTAINTY[0 , th] = f1score_high
            FSCORE_AUDIT[0 , th] = f1score_audit
            AUDIT_THRESHOLD[0 , th] = uncertainty_threshold
            
        accuracy, f1score, recall, precission, conf_mat = compute_metrics(y_test.astype('int'), Probs.astype('int'))

        if args.create_classification_map and len(Thresholds) == 1:
            Classification_map, _, _ = Classification_Maps(Probs, y_test, central_pixels_coordinates_ts_, hit_map)
            Classification_map[mask_original == 4] = brown_colour
            plt.imsave(os.path.join(save_path,'Classification_map.png'), Classification_map)

        TP = conf_mat[1 , 1]
        FP = conf_mat[0 , 1]
        TN = conf_mat[0 , 0]
        FN = conf_mat[1 , 0]
        numerator = TP + FP

        denominator = TN + FN + FP + TP

        Alert_area = 100*(numerator/denominator)
        print(f1score)
        ACCURACY[0 , th] = accuracy
        FSCORE[0 , th] = f1score
        RECALL[0 , th] = recall
        PRECISSION[0 , th] = precission
        CONFUSION_MATRIX[: , : , th] = conf_mat
        ALERT_AREA[0 , th] = Alert_area

    print('Accuracy')
    print(ACCURACY)
    print('Fscore')
    print(FSCORE)
    print('Recall')
    print(RECALL)
    print('Precision')
    print(PRECISSION)
    print('Confusion matrix')
    print(CONFUSION_MATRIX[:,:,0])
    print('Alert_area')
    print(ALERT_AREA)
    print('Uncertainty mean')
    print(UNCERTAINTY_MEAN)
    print('FScore Low Uncertainty')
    print(FSCORE_LOW_UNCERTAINTY)
    print('FScore High Uncertainty')
    print(FSCORE_HIGH_UNCERTAINTY)
    print('FScore Audited')
    print(FSCORE_AUDIT)
    print('Audit Threshold')
    print(AUDIT_THRESHOLD)

    return ACCURACY, FSCORE, RECALL, PRECISSION, CONFUSION_MATRIX, ALERT_AREA, UNCERTAINTY_MEAN, FSCORE_LOW_UNCERTAINTY, FSCORE_HIGH_UNCERTAINTY, FSCORE_AUDIT, AUDIT_THRESHOLD

def Metrics_For_Test_Avg(hit_map,
                     reference_t1, reference_t2,
                     Train_tiles, Valid_tiles, Undesired_tiles,
                     args):

    save_path = os.path.join(args.results_dir, args.file)
    print('[*]Defining the initial central patches coordinates...')
    
    mask_final = mask_creation(reference_t1.shape[0], reference_t1.shape[1], args.horizontal_blocks, args.vertical_blocks, Train_tiles, Valid_tiles, Undesired_tiles)

    #mask_final = mask_final_.copy()
    mask_final[mask_final == 1] = 0
    mask_final[mask_final == 3] = 0
    mask_final[mask_final == 2] = 1

    np.save(os.path.join(save_path,'hit_map'), hit_map)

    Probs_init = hit_map
    
    epsilon = 1e-10
    uncertainty_map = -(hit_map * np.log2(hit_map + epsilon) + (1 - hit_map) * np.log2(1 - hit_map + epsilon))
    
    positive_map_init = np.zeros_like(Probs_init)

    reference_t1_copy_ = reference_t1.copy()
    reference_t1_copy_ = reference_t1_copy_ - 1
    reference_t1_copy_[reference_t1_copy_ == -1] = 1
    reference_t1_copy_[reference_t2 == 2] = 0
    mask_f_ = mask_final * reference_t1_copy_
    
    min_array = np.zeros((1 , ))
    Pmax = np.max(Probs_init[mask_f_ == 1])
    probs_list = np.arange(Pmax, 0, -Pmax/(args.Npoints - 1))
    Thresholds = np.concatenate((probs_list , min_array))

    print('Max probability value:')
    print(Pmax)
    print('Thresholds:')
    print(Thresholds)
    # Metrics containers
    ACCURACY = np.zeros((1, len(Thresholds)))
    FSCORE = np.zeros((1, len(Thresholds)))
    RECALL = np.zeros((1, len(Thresholds)))
    PRECISSION = np.zeros((1, len(Thresholds)))
    CONFUSION_MATRIX = np.zeros((2 , 2, len(Thresholds)))
    CLASSIFICATION_MAPS = np.zeros((len(Thresholds), hit_map.shape[0], hit_map.shape[1], 3))
    ALERT_AREA = np.zeros((1 , len(Thresholds)))

    print('[*]The metrics computation has started...')
    #Computing the metrics for each defined threshold
    for th in range(len(Thresholds)):
        print(Thresholds[th])

        positive_map_init = np.zeros_like(hit_map)
        reference_t1_copy = reference_t1.copy()

        threshold = Thresholds[th]
        positive_coordinates = np.transpose(np.array(np.where(Probs_init >= threshold)))
        positive_map_init[positive_coordinates[:,0].astype('int'), positive_coordinates[:,1].astype('int')] = 1

        if args.eliminate_regions:
            positive_map_init_ = skimage.morphology.area_opening(positive_map_init.astype('int'),area_threshold = args.area_avoided, connectivity=1)
            eliminated_samples = positive_map_init - positive_map_init_
        else:
            eliminated_samples = np.zeros_like(hit_map)

        reference_t1_copy = reference_t1_copy + eliminated_samples
        reference_t1_copy[reference_t1_copy == 2] = 1
        reference_t1_copy = reference_t1_copy - 1
        reference_t1_copy[reference_t1_copy == -1] = 1
        reference_t1_copy[reference_t2 == 2] = 0
        mask_f = mask_final * reference_t1_copy

        #central_pixels_coordinates_ts, y_test = Central_Pixel_Definition_For_Test(mask_final_, reference_t1_copy, reference_t2, args.patches_dimension, 1, 'metrics')
        central_pixels_coordinates_ts_ = np.transpose(np.array(np.where(mask_f == 1)))
        
        y_test = reference_t2[central_pixels_coordinates_ts_[:,0].astype('int'), central_pixels_coordinates_ts_[:,1].astype('int')]
        
        Probs = hit_map[central_pixels_coordinates_ts_[:,0].astype('int'), central_pixels_coordinates_ts_[:,1].astype('int')]
        
        Probs[Probs >= Thresholds[th]] = 1
        Probs[Probs <  Thresholds[th]] = 0

        accuracy, f1score, recall, precission, conf_mat = compute_metrics(y_test.astype('int'), Probs.astype('int'))

        #Classification_map, _, _ = Classification_Maps(Probs, y_test, central_pixels_coordinates_ts_, hit_map)

        TP = conf_mat[1 , 1]
        FP = conf_mat[0 , 1]
        TN = conf_mat[0 , 0]
        FN = conf_mat[1 , 0]
        numerator = TP + FP
        denominator = TN + FN + FP + TP
        Alert_area = 100*(numerator/denominator)
        #print(f1score)
        print(precission)
        print(recall)
        ACCURACY[0 , th] = accuracy
        FSCORE[0 , th] = f1score
        RECALL[0 , th] = recall
        PRECISSION[0 , th] = precission
        CONFUSION_MATRIX[: , : , th] = conf_mat
        #CLASSIFICATION_MAPS[th, :, :, :] = Classification_map
        ALERT_AREA[0 , th] = Alert_area
    
    max_idx = np.argmax(FSCORE)
    max_fscore = np.max(FSCORE)
    max_threshold = Thresholds[max_idx]

    #Saving the metrics as npy array
    np.save(os.path.join(save_path,'Accuracy'), ACCURACY)
    np.save(os.path.join(save_path,'Fscore'), FSCORE)
    np.save(os.path.join(save_path,'Recall'), RECALL)
    np.save(os.path.join(save_path,'Precission'), PRECISSION)
    np.save(os.path.join(save_path,'Confusion_matrix'), CONFUSION_MATRIX)
    #np.save(os.path.join(save_path,'Classification_maps'), CLASSIFICATION_MAPS)
    np.save(os.path.join(save_path,'Alert_area'), ALERT_AREA)
    np.save(os.path.join(save_path,'Max_fscore'), max_fscore)
    np.save(os.path.join(save_path,'Max_threshold'), max_threshold)
    np.save(os.path.join(save_path,'Uncertainty_map'), uncertainty_map)

    return ACCURACY, FSCORE, RECALL, PRECISSION, ALERT_AREA, max_threshold, max_idx



def Metrics_For_Uncertainty(hit_map,
                     reference_t1, reference_t2,
                     Train_tiles, Valid_tiles, Undesired_tiles,
                     args):

    print('Computing Metrics_For_Uncertainty')

    uncertainty_path = os.path.join(args.average_results_dir, args.file,'Uncertainty_map.npy')
    
    if not os.path.exists(uncertainty_path):
        raise Exception(f"There is no uncertainty map recorded in: {uncertainty_path}")
    
    uncertainty_map = np.load(uncertainty_path)

    audit_area = [i for i in range(21)]
    
    threshold = 0.5

    save_path = os.path.join(args.results_dir, args.file)
    print('[*]Defining the initial central patches coordinates...')
    #mask_init = mask_creation(reference_t1.shape[0], reference_t1.shape[1], args.horizontal_blocks, args.vertical_blocks, [], [], [])
    mask_final = mask_creation(reference_t1.shape[0], reference_t1.shape[1], args.horizontal_blocks, args.vertical_blocks, Train_tiles, Valid_tiles, Undesired_tiles)

    #mask_final = mask_final_.copy()
    mask_final[mask_final == 1] = 0
    mask_final[mask_final == 3] = 0
    mask_final[mask_final == 2] = 1

    Probs_init = hit_map
    
    # Metrics containers
    # ROWS: different values for audit area
    # 5 COLS: f1score, f1score_low, f1score_high, f1score_audit, uncertainty_threshold
    FSCORE_METRICS = np.zeros((len(audit_area),5))

    print('[*]The metrics computation has started...')

    print('Threshold:')
    print(threshold)

    positive_map_init = np.zeros_like(hit_map)
    reference_t1_copy = reference_t1.copy()
    
    positive_coordinates = np.transpose(np.array(np.where(Probs_init >= threshold)))
    positive_map_init[positive_coordinates[:,0].astype('int'), positive_coordinates[:,1].astype('int')] = 1

    if args.eliminate_regions:
        positive_map_init_ = skimage.morphology.area_opening(positive_map_init.astype('int'),area_threshold = args.area_avoided, connectivity=1)
        eliminated_samples = positive_map_init - positive_map_init_
    else:
        eliminated_samples = np.zeros_like(hit_map)

    reference_t1_copy = reference_t1_copy + eliminated_samples
    reference_t1_copy[reference_t1_copy == 2] = 1

    reference_t1_copy = reference_t1_copy - 1
    reference_t1_copy[reference_t1_copy == -1] = 1
    reference_t1_copy[reference_t2 == 2] = 0
    mask_f = mask_final * reference_t1_copy

    #central_pixels_coordinates_ts, y_test = Central_Pixel_Definition_For_Test(mask_final_, reference_t1_copy, reference_t2, args.patches_dimension, 1, 'metrics')
    central_pixels_coordinates_ts_ = np.transpose(np.array(np.where(mask_f == 1)))
    
    y_test = reference_t2[central_pixels_coordinates_ts_[:,0].astype('int'), central_pixels_coordinates_ts_[:,1].astype('int')]

    #print(np.shape(central_pixels_coordinates_ts))
    Probs = hit_map[central_pixels_coordinates_ts_[:,0].astype('int'), central_pixels_coordinates_ts_[:,1].astype('int')]
    
    Probs[Probs >= threshold] = 1
    Probs[Probs < threshold] = 0
    
    uncertaint_map_ = uncertainty_map[central_pixels_coordinates_ts_[:,0].astype('int'), central_pixels_coordinates_ts_[:,1].astype('int')]
    uncertaint_map_ = np.maximum(uncertaint_map_, 0)
    
    for index, aa in enumerate(audit_area):
        uncertainty_threshold = np.percentile(uncertaint_map_, 100.0 - aa)
        mask_uncertainty = uncertaint_map_ >= uncertainty_threshold
        high_uncertainty_mask = np.where(mask_uncertainty, 1, 0)
        
        f1score, f1score_low, f1score_high, f1score_audit = compute_f1_uncertainty(y_test.astype('int'), Probs.astype('int'), high_uncertainty_mask)
        
        FSCORE_METRICS[index,0] = f1score
        FSCORE_METRICS[index,1] = f1score_low
        FSCORE_METRICS[index,2] = f1score_high
        FSCORE_METRICS[index,3] = f1score_audit
        FSCORE_METRICS[index,4] = uncertainty_threshold
        
    np.save(os.path.join(save_path,'fscore_metrics'), FSCORE_METRICS)

    return FSCORE_METRICS