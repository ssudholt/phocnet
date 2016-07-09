'''
Created on Jul 8, 2016

@author: ssudholt
'''
from argparse import ArgumentParser
import logging
import os

class BasePHOCNetExperiment(object):
    '''
    Base class for all PHOCNet experiments
    '''

    def __init__(self, dataset_dir, train_annotation_file, test_annotation_file, 
                 cnn_proto_path, solver_proto_path, n_train_images, 
                 lmdb_dir, save_net_path, recreate_lmdbs):
        '''
        The constructor
        
        Args:
            dataset_dir (str): the location of the document images for the given dataset
            train_annotation_file (str): the absolute path to the READ-style annotation file for the training samples
            test_annotation_file (str): the absolute path to the READ-style annotation file for the test samples
            cnn_proto_path (str): absolute path where to save the Caffe CNN model protobuffer file
            solver_proto_path (str): absolute path where to save the Caffe Solver protobuffer file
            n_train_images (int): the total number of training images to be used
            lmdb_dir (str): directory to save the LMDB files into
            save_net_path (str): absolute path where to save the trained PHOCNet
            recreate_lmdbs (bool): whether to delete and recompute existing LMDBs
        '''
        # store the parameters
        self.dataset_dir = dataset_dir
        self.train_annotation_file = train_annotation_file
        self.test_annotation_file = test_annotation_file
        self.cnn_proto_path = cnn_proto_path
        self.solver_proto_path = solver_proto_path
        self.n_train_images = n_train_images
        self.lmdb_dir = lmdb_dir
        self.save_net_path = save_net_path
        self.recreate_lmdbs = recreate_lmdbs
        
        # set up the logging
        logging_format = '[%(asctime)-19s, %(name)s] %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=logging_format)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def train_phocnet(self):
        # --- Step 1: check if we need to create the LMDBs
        train_word_images_lmdb_path = os.path.join(self.lmdb_dir, '%s_train_word_images_lmdb')
        train_phoc_lmdb_path = os.path.join(self.lmdb_dir, '%s_train_phoc_lmdb')
        test_word_images_lmdb_path = os.path.join(self.lmdb_dir, '%s_test_word_images_lmdb')
        test_phoc_lmdb_path = os.path.join(self.lmdb_dir, '%s_test_phoc_lmdb')
        
        # --- Step 2: create the proto files
        
        # --- Step 3: train the PHOCNet
        
        
    

class PHOCNetExperimentArgumentParser(ArgumentParser):
    def __init__(self):
        # IO parameters
        self.add_argument('--save_net_path', '-snp', action='store', type=str,
                          help='Absolute path where to save the final PHOCNet. If unspecified, the net is not saved after training')
        self.add_argument('--recreate_lmdbs', '-rl', action='store_true', default=False,
                          help='Flag indicating to delete existing LMDBs for this dataset and recompute them.')
    def parse_and_run(self, experiment_driver_class):
        args = vars(self.parse_args())        
        experiment_instance = experiment_driver_class(**args)
        experiment_instance.train_phocnet()