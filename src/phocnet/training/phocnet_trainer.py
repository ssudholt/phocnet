'''
Created on Aug 29, 2016

@author: ssudholt
'''
import logging
import os
import time

import caffe
import numpy as np
from skimage.transform import resize

from phocnet.attributes.phoc import build_phoc, unigrams_from_word_list,\
    get_most_common_n_grams
from phocnet.caffe.model_proto_generator import ModelProtoGenerator
from phocnet.caffe.solver_proto_generator import generate_solver_proto
from phocnet.caffe.lmdb_creator import CaffeLMDBCreator
from phocnet.caffe.augmentation import AugmentationCreator
from phocnet.evaluation.time import convert_secs2HHMMSS
from phocnet.evaluation.cnn import calc_map_from_cnn_features
from phocnet.io.xml_io import XMLReader
from phocnet.io.files import save_prototxt, write_list
from phocnet.io.context_manager import Suppressor
from phocnet.numpy.numpy_helper import NumpyHelper

class PHOCNetTrainer(object):
    '''
    Driver class for all PHOCNet experiments
    '''

    def __init__(self, doc_img_dir, train_annotation_file, test_annotation_file, 
                 proto_dir, n_train_images, lmdb_dir, save_net_dir, 
                 phoc_unigram_levels, recreate_lmdbs, gpu_id, learning_rate, momentum, 
                 weight_decay, batch_size, test_interval, display, max_iter, step_size, 
                 gamma, debug_mode, metric, annotation_delimiter, use_lower_case_only,
                 use_bigrams):
        '''
        The constructor
        
        Args:
            doc_img_dir (str): the location of the document images for the given dataset
            train_annotation_file (str): the absolute path to the READ-style annotation file for the training samples
            test_annotation_file (str): the absolute path to the READ-style annotation file for the test samples
            proto_dir (str): absolute path where to save the Caffe protobuffer files
            n_train_images (int): the total number of training images to be used
            lmdb_dir (str): directory to save the LMDB files into
            save_net_dir (str): directory where to save the trained PHOCNet
            phoc_unigrams_levels (list of int): the list of unigram levels
            recreate_lmdbs (bool): whether to delete and recompute existing LMDBs
            debug_mode (bool): flag indicating to run this experiment in debug mode
            metric (str): metric for comparing the PHOCNet output during test
            annotation_delimiter (str): delimiter for the annotation in the XML files
            use_lower_case_only (bool): convert annotation to lower case before creating LMDBs
            use_bigrams (bool): if true, the PHOC predicted from the net contains bigrams
            
            gpu_id (int): the ID of the GPU to use
            learning_rate (float): the learning rate to be used in training
            momentum (float): the SGD momentum to be used in training
            weight_decay (float): the SGD weight decay to be used in training
            batch_size (int): the number of images to be used in a mini batch
            test_interval (int): the number of steps after which to evaluate the PHOCNet during training
            display (int): the number of iterations after which to show the training net loss
            max_iter (int): the maximum number of SGD iterations
            step_size (int): the number of iterations after which to reduce the learning rate
            gamma (float): the factor to multiply the step size with after step_size iterations
        '''
        # store the experiment parameters
        self.doc_img_dir = doc_img_dir
        self.train_annotation_file = train_annotation_file
        self.test_annotation_file = test_annotation_file
        self.proto_dir = proto_dir
        self.n_train_images = n_train_images
        self.lmdb_dir = lmdb_dir
        self.save_net_dir = save_net_dir
        self.phoc_unigram_levels = phoc_unigram_levels
        self.recreate_lmdbs = recreate_lmdbs
        self.debug_mode = debug_mode
        self.metric = metric
        self.annotation_delimiter = annotation_delimiter
        self.use_lower_case_only = use_lower_case_only
        self.use_bigrams = use_bigrams
        
        # store the Caffe parameters
        self.gpu_id = gpu_id
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.test_interval = test_interval
        self.display = display
        self.max_iter = max_iter
        self.step_size = step_size
        self.gamma = gamma        
        
        # misc members for training/evaluation
        if self.gpu_id is not None:
            self.solver_mode = 'GPU'
        else:
            self.solver_mode = 'CPU'
        self.min_image_width_height = 26
        self.epoch_map = None
        self.test_iter = None
        self.dataset_name = None
        
        # set up the logging
        logging_format = '[%(asctime)-19s, %(name)s] %(message)s'
        if self.debug_mode:
            logging_level = logging.DEBUG
        else:
            logging_level = logging.INFO
        logging.basicConfig(level=logging_level, format=logging_format)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def train_phocnet(self):
        self.logger.info('--- Running PHOCNet Training ---')
        # --- Step 1: check if we need to create the LMDBs
        # load the word lists
        xml_reader = XMLReader(make_lower_case=self.use_lower_case_only)
        self.dataset_name, train_list, test_list = xml_reader.load_train_test_xml(train_xml_path=self.train_annotation_file, 
                                                                                  test_xml_path=self.test_annotation_file, 
                                                                                  img_dir=self.doc_img_dir)
        phoc_unigrams = unigrams_from_word_list(word_list=train_list, split_character=self.annotation_delimiter)
        self.logger.info('PHOC unigrams: %s', ' '.join(phoc_unigrams))
        self.test_iter = len(test_list)
        self.logger.info('Using dataset \'%s\'', self.dataset_name)
        
        # check if we need to create LMDBs
        lmdb_prefix = '%s_nti%d_pul%s' % (self.dataset_name, self.n_train_images,
                                          '-'.join([str(elem) for elem in self.phoc_unigram_levels]))
        train_word_images_lmdb_path = os.path.join(self.lmdb_dir, '%s_train_word_images_lmdb' % lmdb_prefix)
        train_phoc_lmdb_path = os.path.join(self.lmdb_dir, '%s_train_phocs_lmdb' % lmdb_prefix)
        test_word_images_lmdb_path = os.path.join(self.lmdb_dir, '%s_test_word_images_lmdb' % lmdb_prefix)
        test_phoc_lmdb_path = os.path.join(self.lmdb_dir, '%s_test_phocs_lmdb' % lmdb_prefix)
        lmdbs_exist = (os.path.exists(train_word_images_lmdb_path),
                       os.path.exists(train_phoc_lmdb_path),
                       os.path.exists(test_word_images_lmdb_path),
                       os.path.exists(test_phoc_lmdb_path))
                        
        if self.use_bigrams:
            n_bigrams = 50
            bigrams = get_most_common_n_grams(words=[word.get_transcription() 
                                                     for word in train_list], 
                                              num_results=n_bigrams, n=2)
            bigram_levels = [2]
        else:       
            n_bigrams = 0         
            bigrams = None
            bigram_levels = None        
        if not np.all(lmdbs_exist) or self.recreate_lmdbs:     
            self.logger.info('Creating LMDBs...')  
                    
                
            train_phocs = build_phoc(words=[word.get_transcription() for word in train_list], 
                                     phoc_unigrams=phoc_unigrams, unigram_levels=self.phoc_unigram_levels,
                                     phoc_bigrams=bigrams, bigram_levels=bigram_levels,
                                     split_character=self.annotation_delimiter,
                                     on_unknown_unigram='warn')
            test_phocs = build_phoc(words=[word.get_transcription() for word in test_list],
                                    phoc_unigrams=phoc_unigrams, unigram_levels=self.phoc_unigram_levels,
                                    phoc_bigrams=bigrams, bigram_levels=bigram_levels,
                                    split_character=self.annotation_delimiter,
                                    on_unknown_unigram='warn')
            self._create_train_test_phocs_lmdbs(train_list=train_list, train_phocs=train_phocs, 
                                                test_list=test_list, test_phocs=test_phocs,
                                                train_word_images_lmdb_path=train_word_images_lmdb_path,
                                                train_phoc_lmdb_path=train_phoc_lmdb_path,
                                                test_word_images_lmdb_path=test_word_images_lmdb_path,
                                                test_phoc_lmdb_path=test_phoc_lmdb_path)
        else:
            self.logger.info('Found LMDBs...')
        
        # --- Step 2: create the proto files
        self.logger.info('Saving proto files...')
        # prepare the output paths
        train_proto_path = os.path.join(self.proto_dir, 'train_phocnet_%s.prototxt' % self.dataset_name)
        test_proto_path = os.path.join(self.proto_dir, 'test_phocnet_%s.prototxt' % self.dataset_name)
        solver_proto_path = os.path.join(self.proto_dir, 'solver_phocnet_%s.prototxt' % self.dataset_name)
        
        # generate the proto files
        n_attributes = np.sum(self.phoc_unigram_levels)*len(phoc_unigrams)
        if self.use_bigrams:
            n_attributes += np.sum(bigram_levels)*n_bigrams
        mpg = ModelProtoGenerator(initialization='msra', use_cudnn_engine=self.gpu_id is not None)        
        train_proto = mpg.get_phocnet(word_image_lmdb_path=train_word_images_lmdb_path, phoc_lmdb_path=train_phoc_lmdb_path, 
                                      phoc_size=n_attributes, 
                                      generate_deploy=False)
        test_proto = mpg.get_phocnet(word_image_lmdb_path=test_word_images_lmdb_path, phoc_lmdb_path=test_phoc_lmdb_path, 
                                     phoc_size=n_attributes, generate_deploy=False)
        solver_proto = generate_solver_proto(train_net=train_proto_path, test_net=test_proto_path,
                                             base_lr=self.learning_rate, momentum=self.momentum, display=self.display,
                                             lr_policy='step', gamma=self.gamma, stepsize=self.step_size,
                                             solver_mode=self.solver_mode, iter_size=self.batch_size, max_iter=self.max_iter,
                                             average_loss=self.display, test_iter=self.test_iter, test_interval=self.test_interval,
                                             weight_decay=self.weight_decay)
        # save the proto files
        save_prototxt(file_path=train_proto_path, proto_object=train_proto, header_comment='Train PHOCNet %s' % self.dataset_name)
        save_prototxt(file_path=test_proto_path, proto_object=test_proto, header_comment='Test PHOCNet %s' % self.dataset_name)
        save_prototxt(file_path=solver_proto_path, proto_object=solver_proto, header_comment='Solver PHOCNet %s' % self.dataset_name)
        
        # --- Step 3: train the PHOCNet
        self.logger.info('Starting SGD...')
        self._run_sgd(solver_proto_path=solver_proto_path)

    def pretrain_callback(self, solver):
        '''
        Method called before starting the training
        '''        
        # init numpy arrays for mAP results        
        epochs = self.max_iter/self.test_interval
        self.epoch_map = np.zeros(epochs+1)
        self.epoch_map[0], _ = calc_map_from_cnn_features(solver=solver, 
                                                          test_iterations=self.test_iter, 
                                                          metric=self.metric)
        self.logger.info('mAP: %f', self.epoch_map[0])
    
    def test_callback(self, solver, epoch):
        '''
        Method called every self.test_interval iterations during training
        '''
        self.logger.info('Evaluating CNN after %d steps:', epoch*solver.param.test_interval)
        self.epoch_map[epoch+1], _ = calc_map_from_cnn_features(solver=solver, 
                                                                test_iterations=self.test_iter, 
                                                                metric=self.metric)
        self.logger.info('mAP: %f', self.epoch_map[epoch+1])
    
    def posttrain_callback(self, solver):
        '''
        Method called after finishing the training
        '''
        # if self.save_net is not None, save the PHOCNet to the desired location
        if self.save_net_dir is not None:
            filename = 'phocnet_%s_nti%d_pul%s.binaryproto' % (self.dataset_name, self.n_train_images,
                                                               '-'.join([str(elem) for elem in self.phoc_unigram_levels]))
            solver.net.save(os.path.join(self.save_net_dir, filename))
    
    def _create_train_test_phocs_lmdbs(self, train_list, train_phocs, test_list, test_phocs, 
                                       train_word_images_lmdb_path, train_phoc_lmdb_path,
                                       test_word_images_lmdb_path, test_phoc_lmdb_path):
        start_time = time.time()        
        # --- TRAIN IMAGES
        # find all unique transcriptions and the label map...
        _, transcription_map = self.__get_unique_transcriptions_and_labelmap(train_list, test_list)
        # get the numeric training labels plus a random order to insert them into
        # create the numeric labels and counts
        train_labels = np.array([transcription_map[word.get_transcription()] for word in train_list])
        unique_train_labels, counts = np.unique(train_labels, return_counts=True)
        # find the number of images that should be present for training per class
        n_images_per_class = self.n_train_images/unique_train_labels.shape[0] + 1
        # create randomly shuffled numbers for later use as keys
        random_indices = list(xrange(n_images_per_class*unique_train_labels.shape[0]))
        np.random.shuffle(random_indices)
                
        
        #set random limits for affine transform
        random_limits = (0.8, 1.1)
        n_rescales = 0
        
        # loading should be done in gray scale
        load_grayscale = True
        
        # create train LMDB  
        self.logger.info('Creating Training LMDB (%d total word images)', len(random_indices))      
        lmdb_creator = CaffeLMDBCreator()
        lmdb_creator.open_dual_lmdb_for_write(image_lmdb_path=train_word_images_lmdb_path, 
                                              additional_lmdb_path=train_phoc_lmdb_path,
                                              create=True)
        for cur_label, count in zip(unique_train_labels, counts):
            # find the words for the current class label and the
            # corresponding PHOC            
            cur_word_indices = np.where(train_labels == cur_label)[0]  
            cur_transcription = train_list[cur_word_indices[0]].get_transcription()
            cur_phoc = NumpyHelper.get_unique_rows(train_phocs[cur_word_indices])
            # unique rows should only return one specific PHOC
            if cur_phoc.shape[0] != 1:
                raise ValueError('Extracted more than one PHOC for label %d' % cur_label)
            cur_phoc = np.atleast_3d(cur_phoc).transpose((2,0,1)).astype(np.uint8)
                      
            # if there are to many images for the current word image class, 
            # draw from them and cut the rest off
            if count > n_images_per_class:
                np.random.shuffle(cur_word_indices)
                cur_word_indices = cur_word_indices[:n_images_per_class]
            # load the word images
            cur_word_images = []            
            for idx in cur_word_indices:                
                img = train_list[idx].get_word_image(gray_scale=load_grayscale)  
                # check image size
                img, resized = self.__check_size(img)
                n_rescales += int(resized)
                
                # append to the current word images and
                # put into LMDB
                cur_word_images.append(img)
                key = '%s_%s' % (str(random_indices.pop()).zfill(8), cur_transcription.encode('ascii', 'ignore'))                
                lmdb_creator.put_dual(img_mat=np.atleast_3d(img).transpose((2,0,1)).astype(np.uint8), 
                                      additional_mat=cur_phoc, label=cur_label, key=key)
                            
            # extract the extra augmented images
            # the random limits are the maximum percentage
            # that the destination point may deviate from the reference point
            # in the affine transform            
            if len(cur_word_images) < n_images_per_class:
                # create the warped images
                inds = np.random.randint(len(cur_word_images), size=n_images_per_class - len(cur_word_images))                
                for ind in inds:
                    aug_img = AugmentationCreator.create_affine_transform_augmentation(img=cur_word_images[ind], random_limits=random_limits)
                    aug_img = np.atleast_3d(aug_img).transpose((2,0,1)).astype(np.uint8)
                    key = '%s_%s' % (str(random_indices.pop()).zfill(8), cur_transcription.encode('ascii', 'ignore'))
                    lmdb_creator.put_dual(img_mat=aug_img, additional_mat=cur_phoc, label=cur_label, key=key)
        # wrap up training LMDB creation
        if len(random_indices) != 0:
            raise ValueError('Random Indices are not empty, something went wrong during training LMDB creation')
        lmdb_creator.finish_creation()
        # write the label map to the LMDBs as well        
        write_list(file_path=train_word_images_lmdb_path + '/label_map.txt', 
                   line_list=['%s %s' % elem for elem in transcription_map.items()])
        write_list(file_path=train_phoc_lmdb_path + '/label_map.txt', 
                   line_list=['%s %s' % elem for elem in transcription_map.items()])
        self.logger.info('Finished processing train words (took %s, %d rescales)', convert_secs2HHMMSS(time.time() - start_time), n_rescales)
        
        # --- TEST IMAGES
        self.logger.info('Creating Test LMDB (%d total word images)', len(test_list))
        n_rescales = 0
        start_time = time.time()
        lmdb_creator.open_dual_lmdb_for_write(image_lmdb_path=test_word_images_lmdb_path, additional_lmdb_path=test_phoc_lmdb_path, 
                                              create=True, label_map=transcription_map)
        for word, phoc in zip(test_list, test_phocs): 
            if word.get_transcription() not in transcription_map:
                transcription_map[word.get_transcription()] = len(transcription_map)
            img = word.get_word_image(gray_scale=load_grayscale)
            img, resized = self.__check_size(img)
            if img is None:
                    self.logger.warning('!WARNING! Found image with 0 width or height!')
            else:
                n_rescales += int(resized)
                img = np.atleast_3d(img).transpose((2,0,1)).astype(np.uint8)
                phoc_3d = np.atleast_3d(phoc).transpose((2,0,1)).astype(np.uint8)
                lmdb_creator.put_dual(img_mat=img, additional_mat=phoc_3d, label=transcription_map[word.get_transcription()])
        lmdb_creator.finish_creation()
        write_list(file_path=test_word_images_lmdb_path + '/label_map.txt', 
                   line_list=['%s %s' % elem for elem in transcription_map.items()])
        write_list(file_path=test_phoc_lmdb_path + '/label_map.txt', 
                   line_list=['%s %s' % elem for elem in transcription_map.items()])
        self.logger.info('Finished processing test words (took %s, %d rescales)', convert_secs2HHMMSS(time.time() - start_time), n_rescales)
    
    def __check_size(self, img):
        '''
        checks if the image accords to the minimum size requirements
        
        Returns:
            tuple (img, bool):
                 img: the original image if the image size was ok, a resized image otherwise
                 bool: flag indicating whether the image was resized
        '''
        if np.amin(img.shape[:2]) < self.min_image_width_height:
            if np.amin(img.shape[:2]) == 0:
                return None, False
            scale = float(self.min_image_width_height+1)/float(np.amin(img.shape[:2]))
            new_shape = (int(scale*img.shape[0]), int(scale*img.shape[1]))
            new_img = resize(image=img, output_shape=new_shape)
            return new_img, True
        else:
            return img, False 
    
    def __get_unique_transcriptions_and_labelmap(self, train_list, test_list):
        '''
        Returns a list of unique transcriptions for the given train and test lists
        and creates a dictionary mapping transcriptions to numeric class labels.
        '''
        unique_transcriptions = [word.get_transcription() for word in train_list]
        unique_transcriptions.extend([word.get_transcription() for word in test_list])
        unique_transcriptions = list(set(unique_transcriptions))
        transcription_map = dict((k,v) for v,k in enumerate(unique_transcriptions))
        return unique_transcriptions, transcription_map        
    
    def _run_sgd(self, solver_proto_path):
        '''
        Starts the SGD training of the PHOCNet
        
        Args:
            solver_proto_path (str): the absolute path to the solver protobuffer file to use
        '''
        # Set CPU/GPU mode for solver training
        if self.gpu_id != None:
            self.logger.info('Setting Caffe to GPU mode using device %d', self.gpu_id)
            caffe.set_mode_gpu()
            caffe.set_device(self.gpu_id)
        else:
            self.logger.info('Setting Caffe to CPU mode')
            caffe.set_mode_cpu()
        
        # Create SGD solver
        self.logger.info('Using solver protofile at %s', solver_proto_path)
        solver = self.__get_solver(solver_proto_path)        
        epochs = self.max_iter/self.test_interval
        
        # run test on the net before training
        self.logger.info('Running pre-train evaluation')
        self.pretrain_callback(solver=solver)
        
        # run the training
        self.logger.info('Finished Setup, running SGD')        
        for epoch in xrange(epochs):
            # run training until we want to test
            self.__solver_step(solver, self.test_interval)
            
            # run test callback after test_interval iterations
            self.logger.debug('Running test evaluation')
            self.test_callback(solver=solver, epoch=epoch)
        # if we have iterations left to compute, do so
        iters_left = self.max_iter % self.test_interval
        if iters_left > 0:
            self.__solver_step(solver, iters_left)
            
        # run post train callback
        self.logger.info('Running post-train evaluation')
        self.posttrain_callback(solver=solver)
        # return the solver
        return solver
    
    def __solver_step(self, solver, steps):
        '''
        Runs Caffe solver suppressing Caffe output if necessary
        '''
        if not self.debug_mode:
            with Suppressor():
                solver.step(steps)
        else:
            solver.step(steps)
    
    def __get_solver(self, solver_proto_path):
        '''
        Returns a caffe.SGDSolver for the given protofile path,
        ignoring Caffe command line chatter if debug mode is not set
        to True.
        '''
        if not self.debug_mode:
            # disable Caffe init chatter when not in debug
            with Suppressor():
                return caffe.SGDSolver(solver_proto_path)
        else:
            return caffe.SGDSolver(solver_proto_path)