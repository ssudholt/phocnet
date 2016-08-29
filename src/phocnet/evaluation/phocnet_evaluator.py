'''
Created on Aug 29, 2016

@author: ssudholt
'''
import logging 
import os

import caffe
import numpy as np
from skimage.transform import resize

from phocnet.caffe.model_proto_generator import ModelProtoGenerator
from phocnet.io.xml_io import XMLReader
from phocnet.io.context_manager import Suppressor
from phocnet.attributes.phoc import unigrams_from_word_list, build_phoc,\
    get_most_common_n_grams
from phocnet.io.files import write_list
from phocnet.evaluation.retrieval import map_from_feature_matrix,\
    map_from_query_test_feature_matrices
from phocnet.io import word_list

class PHOCNetEvaluation(object):
    def __init__(self):
        logging_format = '[%(asctime)-19s, %(name)s] %(message)s'
        logging.basicConfig(level=logging.INFO, format=logging_format)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def predict_and_save_phocs(self,phocnet_bin_path, train_xml_file, test_xml_file, 
                               gpu_id, debug_mode, doc_img_dir, phoc_unigram_levels, 
                               deploy_proto_path, phoc_size, output_dir, no_bigrams,
                               annotation_delimiter):
        self.logger.info('--- Predict and save PHOCS ---')
        train_list = self._load_word_list_from_xml(train_xml_file, doc_img_dir)
        test_list = self._load_word_list_from_xml(test_xml_file, doc_img_dir)
        
        phoc_unigrams = unigrams_from_word_list(word_list=train_list, split_character=annotation_delimiter)
        phoc_size = np.sum(phoc_unigram_levels)*len(phoc_unigrams)
        if not no_bigrams:
            phoc_size += 100
        
        phocnet = self._load_pretrained_phocnet(phocnet_bin_path, gpu_id, debug_mode, 
                                                deploy_proto_path, phoc_size)
        self.logger.info('Predicting PHOCs for %d test words', len(test_list))
        phocs = self._net_output_for_word_list(word_list=test_list, cnn=phocnet,
                                               suppress_caffe_output=not debug_mode)
        self._save_phocs(phocs, output_dir)
    
    def extract_unigrams(self, word_xml_file, doc_img_dir, annotation_delimiter):
        self.logger.info('--- Extract Unigrams ---')
        self.logger.info('Loading XML file from: %s...', word_xml_file)
        xml_reader = XMLReader(make_lower_case=True)
        dataset_name, word_list = xml_reader.load_word_list_from_READ_xml(xml_filename=word_xml_file, img_dir=doc_img_dir)
        self.logger.info('Found dataset: %s', dataset_name)
        self.logger.info('Saving unigrams to current working directory...')
        phoc_unigrams = unigrams_from_word_list(word_list=word_list, split_character=annotation_delimiter)
        idx_list = ['%d: %s' % elem for elem in enumerate(phoc_unigrams)]
        write_list(file_path='phoc_unigrams.txt', line_list=idx_list)
   
    def eval_qbs(self, phocnet_bin_path, train_xml_file, test_xml_file, phoc_unigram_levels, 
                 gpu_id, debug_mode, doc_img_dir, deploy_proto_path, metric, 
                 annotation_delimiter, no_bigrams):
        self.logger.info('--- Query-by-String Evaluation ---')
        train_list = self._load_word_list_from_xml(train_xml_file, doc_img_dir)
        test_list = self._load_word_list_from_xml(test_xml_file, doc_img_dir)
        
        phoc_unigrams = unigrams_from_word_list(word_list=train_list, split_character=annotation_delimiter)
        phoc_size = np.sum(phoc_unigram_levels)*len(phoc_unigrams)
        if no_bigrams:
            n_bigrams = 0         
            bigrams = None
            bigram_levels = None
        else:
            n_bigrams = 50
            bigrams = get_most_common_n_grams(words=[word.get_transcription() 
                                                     for word in train_list], 
                                              num_results=n_bigrams, n=2)
            bigram_levels = [2]
            phoc_size += 100
        
        # Set CPU/GPU mode
        if gpu_id != None:
            self.logger.info('Setting Caffe to GPU mode using device %d', gpu_id)
            caffe.set_mode_gpu()
            caffe.set_device(gpu_id)
        else:
            self.logger.info('Setting Caffe to CPU mode')
            caffe.set_mode_cpu()
        phocnet = self._load_pretrained_phocnet(phocnet_bin_path, gpu_id, debug_mode, 
                                                deploy_proto_path, phoc_size)
        self.logger.info('Predicting PHOCs for %d test words', len(test_list))
        test_phocs = self._net_output_for_word_list(word_list=test_list, cnn=phocnet, 
                                                    suppress_caffe_output=not debug_mode)
        test_strings = [word.get_transcription() for word in test_list] 
        qry_strings = list(sorted(set(test_strings)))
        qry_phocs = build_phoc(words=qry_strings, phoc_unigrams=phoc_unigrams, unigram_levels=phoc_unigram_levels, 
                               split_character=annotation_delimiter, phoc_bigrams=bigrams, bigram_levels=bigram_levels)
        self.logger.info('Calculating mAP...')
        mean_ap, _ = map_from_query_test_feature_matrices(query_features=qry_phocs, test_features=test_phocs, query_labels=qry_strings, 
                                                          test_labels=test_strings, metric=metric, drop_first=False)
        self.logger.info('mAP: %f', mean_ap*100)
    
    def eval_qbe(self, phocnet_bin_path, train_xml_file, test_xml_file, 
                 gpu_id, debug_mode, doc_img_dir, annotation_delimiter, 
                 deploy_proto_path, metric, phoc_unigram_levels, no_bigrams):
        self.logger.info('--- Query-by-Example Evaluation ---')
        train_list = self._load_word_list_from_xml(train_xml_file, doc_img_dir)
        test_list = self._load_word_list_from_xml(test_xml_file, doc_img_dir)
        
        phoc_unigrams = unigrams_from_word_list(word_list=train_list, split_character=annotation_delimiter)
        phoc_size = np.sum(phoc_unigram_levels)*len(phoc_unigrams)
        if not no_bigrams:
            phoc_size += 100
        
        # Set CPU/GPU mode
        if gpu_id != None:
            self.logger.info('Setting Caffe to GPU mode using device %d', gpu_id)
            caffe.set_mode_gpu()
            caffe.set_device(gpu_id)
        else:
            self.logger.info('Setting Caffe to CPU mode')
            caffe.set_mode_cpu()
        phocnet = self._load_pretrained_phocnet(phocnet_bin_path, gpu_id, debug_mode, 
                                                deploy_proto_path, phoc_size)
        self.logger.info('Predicting PHOCs for %d test words', len(test_list))
        phocs = self._net_output_for_word_list(word_list=test_list, cnn=phocnet,
                                               suppress_caffe_output=not debug_mode)
        self.logger.info('Calculating mAP...')
        _, avg_precs = map_from_feature_matrix(features=phocs, labels=[word.get_transcription() for word in test_list], 
                                               metric=metric, drop_first=True)
        self.logger.info('mAP: %f', np.mean(avg_precs[avg_precs > 0])*100)
    
    def _net_output_for_word_list(self, word_list, cnn, 
                                  min_img_width_height=26,input_layer='word_images', 
                                  output_layer='sigmoid', suppress_caffe_output=False):
        output = []
        for idx, word in enumerate(word_list):
            # scale to correct pixel values (0 = background, 1 = text)
            word_img = word.get_word_image().astype(np.float32)
            word_img -= 255.0
            word_img /= -255.0      
                  
            # check size
            if np.amin(word_img.shape[:2]) < min_img_width_height:
                scale = float(min_img_width_height+1)/float(np.amin(word_img.shape[:2]))
                new_shape = (int(scale*word_img.shape[0]), int(scale*word_img.shape[1]))
                word_img = resize(image=word_img, output_shape=new_shape)
            word_img = word_img.reshape((1,1,) + word_img.shape).astype(np.float32)            
            
            # reshape the PHOCNet
            cnn.blobs[input_layer].reshape(*word_img.shape)
            cnn.reshape()
            
            # forward the word image through the PHOCNet
            cnn.blobs[input_layer].data[...] = word_img            
            if suppress_caffe_output:
                with Suppressor():
                    output.append(cnn.forward()[output_layer].flatten())
            else:
                output.append(cnn.forward()[output_layer].flatten())
            if ((idx+1)%100 == 0 or (idx+1) == len(word_list)):
                self.logger.info('    [ %*d / %d ]', len(str(len(word_list))), idx+1, len(word_list))            
        return np.vstack(output)        
        

    def _load_pretrained_phocnet(self, phocnet_bin_path, gpu_id, debug_mode, deploy_proto_path, phoc_size):
        # create a deploy proto file
        self.logger.info('Saving PHOCNet deploy proto file to %s...', deploy_proto_path)
        mpg = ModelProtoGenerator(initialization='msra', use_cudnn_engine=gpu_id is not None)
        proto = mpg.get_phocnet(word_image_lmdb_path=None, phoc_lmdb_path=None, phoc_size=phoc_size, generate_deploy=True)
        with open(deploy_proto_path, 'w') as proto_file:
            proto_file.write(str(proto))
            
        # create the Caffe PHOCNet object
        self.logger.info('Creating PHOCNet...')
        if debug_mode:
            phocnet = caffe.Net(deploy_proto_path, phocnet_bin_path, caffe.TEST)
        else:
            with Suppressor():
                phocnet = caffe.Net(deploy_proto_path, phocnet_bin_path, caffe.TEST)
        return phocnet

    def _load_word_list_from_xml(self, word_xml_file, doc_img_dir):
        self.logger.info('Loading XML file from: %s...', word_xml_file)
        dataset_name, word_list = XMLReader().load_word_list_from_READ_xml(xml_filename=word_xml_file, img_dir=doc_img_dir)
        self.logger.info('Found dataset: %s', dataset_name)
        return word_list

    def _setup_caffe(self, gpu_id):
        if gpu_id != None:
            self.logger.info('Setting Caffe to GPU mode using device %d', gpu_id)
            caffe.set_mode_gpu()
            caffe.set_device(gpu_id)
        else:
            self.logger.info('Setting Caffe to CPU mode')
            caffe.set_mode_cpu()

    def _predict_phocs(self, phocnet_bin_path, word_xml_file, gpu_id, debug_mode, doc_img_dir, 
                      deploy_proto_path, phoc_size):
        self._setup_caffe(gpu_id)
        # load everything
        word_list = self._load_word_list_from_xml(word_xml_file, doc_img_dir)        
        phocnet = self._load_pretrained_phocnet(phocnet_bin_path, gpu_id, debug_mode, 
                                                deploy_proto_path, phoc_size)            
        # compute the PHOCs
        self.logger.info('Predicting PHOCs...')
        phocs = self._net_output_for_word_list(word_list=word_list, cnn=phocnet, 
                                               suppress_caffe_output=not debug_mode)
        return phocs
    
    def _predict_phocs_for_sliding_window(self, net, word, frame_width, step_size, phoc_size,  
                                          padding=True, input_layer_name='word_images', output_layer_name='sigmoid'):
        # load and transform image for PHOCNet
        img = word.get_word_image().astype(np.float32)
        img -= 255
        img /= -255
        # pad if requested
        if padding:
            img = np.pad(array=img, pad_width=((0,0), (frame_width/2,frame_width/2)), mode='constant')
            
        # determine the output mat shape and init the mat
        phoc_mat = np.zeros((len(xrange(0, img.shape[1]-frame_width, step_size)), phoc_size), dtype=np.float32)
        
        # push every frame through the net
        for idx, offset in enumerate(xrange(0, img.shape[1]-frame_width, step_size)):            
            frame = img[:, offset:offset+frame_width]            
            # convert to 4D array for Caffe
            frame = frame.reshape((1,1,) + frame.shape)                             
            # push the frame through the net            
            net.blobs[input_layer_name].reshape(*frame.shape)
            net.reshape()
            net.blobs[input_layer_name].data[...] = frame
            phoc = net.forward()[output_layer_name].flatten()
            phoc_mat[idx] = phoc
        return phoc_mat
    
    def _save_phocs(self, phocs, output_dir):
        self.logger.info('Saving PHOCs as .npy-file...')
        np.save(os.path.join(output_dir, 'phocs.npy'), phocs)
        self.logger.info('Finished') 
