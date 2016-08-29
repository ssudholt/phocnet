'''
This script uses a pretrained PHOCNet to generate the
output for a given test list
'''
import argparse
import logging 
import os

import caffe
import numpy as np
from skimage.transform import resize

from phocnet.caffe.model_proto_generator import ModelProtoGenerator
from phocnet.io.xml_io import XMLReader
from phocnet.io.context_manager import Suppressor
from phocnet.attributes.phoc import unigrams_from_word_list, build_phoc
from phocnet.io.files import write_list
from phocnet.evaluation.retrieval import map_from_feature_matrix,\
    map_from_query_test_feature_matrices
from phocnet.io import word_list

class PHOCNetEvaluation(object):
    def __init__(self):
        logging_format = '[%(asctime)-19s, %(name)s] %(message)s'
        logging.basicConfig(level=logging.INFO, format=logging_format)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def predict_and_save_phocs(self,phocnet_bin_path, word_xml_file, gpu_id, debug_mode, doc_img_dir, 
                               deploy_proto_path, phoc_size, output_dir):
        self.logger.info('--- Predict and save PHOCS ---')
        phocs = self._predict_phocs(phocnet_bin_path, word_xml_file, gpu_id, debug_mode, doc_img_dir, 
                                    deploy_proto_path, phoc_size)
        self._save_phocs(phocs, output_dir)
    
    def extract_unigrams(self, word_xml_file, doc_img_dir, annotation_delimiter, use_lower_case_only):
        self.logger.info('--- Extract Unigrams ---')
        self.logger.info('Loading XML file from: %s...', word_xml_file)
        xml_reader = XMLReader(make_lower_case=use_lower_case_only)
        dataset_name, word_list = xml_reader.load_word_list_from_READ_xml(xml_filename=word_xml_file, img_dir=doc_img_dir)
        self.logger.info('Found dataset: %s', dataset_name)
        self.logger.info('Saving unigrams to current working directory...')
        phoc_unigrams = unigrams_from_word_list(word_list=word_list, split_character=annotation_delimiter)
        idx_list = ['%d: %s' % elem for elem in enumerate(phoc_unigrams)]
        write_list(file_path='phoc_unigrams.txt', line_list=idx_list)
    
    def predict_and_save_sliding_window(self, phocnet_bin_path, word_xml_file, gpu_id, debug_mode, doc_img_dir, 
                                        deploy_proto_path, phoc_size, output_dir, frame_width, step_size, no_padding):
        self.logger.info('--- Predict PHOCS for Sliding Window ---')
        self._setup_caffe(gpu_id)
        phocnet = self._load_pretrained_phocnet(phocnet_bin_path, gpu_id, 
                                                debug_mode, deploy_proto_path, phoc_size)
        word_list = self._load_word_list_from_xml(word_xml_file, doc_img_dir)
        
        # predict the PHOCS
        self.logger.info('Predicting sliding window PHOCS...')
        for idx, word in enumerate(word_list):
            phocs = self._predict_phocs_for_sliding_window(net=phocnet, word=word, frame_width=frame_width, 
                                                           step_size=step_size, phoc_size=phoc_size, padding=not no_padding)
            file_name = '%s_%d.npy' % (word.get_page(), word.get_id_on_page())
            np.save(os.path.join(output_dir, file_name), phocs)
            if (idx +1) % 100 == 0 or (idx+1) == len(word_list):
                self.logger.info('   [ %*d / %d ]', len(str(len(word_list))), idx+1, len(word_list))
   
    def eval_qbs(self, phocnet_bin_path, train_xml_file, test_xml_file, phoc_unigram_levels, 
                 gpu_id, debug_mode, doc_img_dir, deploy_proto_path, phoc_size, metric, 
                 annotation_delimiter):
        self.logger.info('--- Query-by-String Evaluation ---')
        train_list = self._load_word_list_from_xml(train_xml_file, doc_img_dir)
        test_list = self._load_word_list_from_xml(test_xml_file, doc_img_dir)
        phoc_unigrams = unigrams_from_word_list(word_list=train_list, split_character=annotation_delimiter)
        phocnet = self._load_pretrained_phocnet(phocnet_bin_path, gpu_id, debug_mode, 
                                                deploy_proto_path, phoc_size)
        test_phocs = self._net_output_for_word_list(word_list=test_list, cnn=phocnet, 
                                                    suppress_caffe_output=not debug_mode)
        test_strings = [word.get_transcription() for word in word_list] 
        qry_strings = list(sorted(set(test_strings)))
        qry_phocs = build_phoc(words=qry_strings, phoc_unigrams=phoc_unigrams, unigram_levels=phoc_unigram_levels, 
                               split_character=phoc_unigram_levels)
        self.logger.info('Calculating mAP...')
        mean_ap, _ = map_from_query_test_feature_matrices(query_features=qry_phocs, test_features=test_phocs, query_labels=qry_strings, 
                                                          test_labels=test_strings, metric=metric, drop_first=False)
        self.logger.info('mAP: %f', mean_ap*100)
    
    def eval_qbe(self, phocnet_bin_path, word_xml_file, gpu_id, debug_mode, doc_img_dir, 
                 deploy_proto_path, phoc_size, metric):
        self.logger.info('--- Query-by-Example Evaluation ---')
        word_list = self._load_word_list_from_xml(word_xml_file, doc_img_dir)
        phocnet = self._load_pretrained_phocnet(phocnet_bin_path, gpu_id, debug_mode, 
                                                deploy_proto_path, phoc_size)
        self.logger.info('Predicting PHOCs for %d test words', len(word_list))
        phocs = self._net_output_for_word_list(word_list=word_list, cnn=phocnet,
                                               suppress_caffe_output=not debug_mode)
        self.logger.info('Calculating mAP...')
        _, avg_precs = map_from_feature_matrix(features=phocs, labels=[word.get_transcription() for word in word_list], 
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

if __name__ == '__main__':
    eval = PHOCNetEvaluation()
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers()
    calc_phocs_parser = subparser.add_parser('calc-phocs', help='Predicts PHOCs from a pretrained CNN and saves them')
    extract_unigrams_parser = subparser.add_parser('extract-unigrams', help='Finds the unigrams for a given XML and saves them')
    qbe_parser = subparser.add_parser('qbe', help='Evaluates the supplied CNN in a Query-by-Example scenario (for protocol see Sudholt 2016 - PHOCNet)')
    qbs_parser = subparser.add_parser('qbs', help='Evaluates the supplied CNN in a Query-by-String scenario (for protocol see Sudholt 2016 - PHOCNet)')
    sliding_window_parser = subparser.add_parser('sliding-window', help='Predicts PHOCs for a sliding window given a pretrained CNN and saves them')
    
    # --- calc-phocs
    calc_phocs_parser.add_argument('--phocnet_bin_path', action='store', type=str, required=True,
                                   help='Path to a pretrained PHOCNet binary protofile.')
    calc_phocs_parser.add_argument('--word_xml_file', action='store', type=str, required=True,
                                   help='The READ-style XML file of words for which to predict the PHOCs')
    calc_phocs_parser.add_argument('--doc_img_dir', action='store', type=str, required=True,
                                   help='The location of the document images.')
    calc_phocs_parser.add_argument('--gpu_id', '-gpu', action='store', type=int, 
                                   help='The ID of the GPU to use. If not specified, prediction is run in CPU mode.')
    calc_phocs_parser.add_argument('--debug_mode', '-dm', action='store_true', default=False,
                                   help='Flag indicating to run the PHOCNet training in debug mode.')
    calc_phocs_parser.add_argument('--phoc_size', '-ps', action='store', type=int, default=540,
                                   help='The size of the PHOC to be used. Default: 540')
    calc_phocs_parser.add_argument('--output_dir', '-od', action='store', type=str, default='.',
                                   help='Location where to save the estimated PHOCs. Default: working directory')
    calc_phocs_parser.add_argument('--deploy_proto_path', '-dpp', action='store', type=str, default='/tmp/deploy_phocnet.prototxt',
                                   help='Location where to save the deploy file. Default: /tmp/deploy_phocnet.prototxt')
    calc_phocs_parser.set_defaults(func=eval.predict_and_save_phocs)
    
    # --- extract-unigrams
    extract_unigrams_parser.add_argument('--word_xml_file', action='store', type=str, required=True,
                                         help='The READ-style XML file of words for which to extract the unigrams')
    extract_unigrams_parser.add_argument('--annotation_delimiter', '-ad', action='store', type=str,
                                         help='If the annotation in the XML is separated by special delimiters, it can be specified here.')
    extract_unigrams_parser.add_argument('--doc_img_dir', action='store', type=str, required=True,
                                         help='The location of the document images.')  
    extract_unigrams_parser.add_argument('--use_lower_case_only', '-ulco', action='store_true', default=False,
                                         help='Flag indicating to convert all annotations from the XML to lower case before proceeding')
    extract_unigrams_parser.set_defaults(func=eval.extract_unigrams)
    
    # --- qbs
    qbs_parser.add_argument('--phocnet_bin_path', action='store', type=str, required=True,
                            help='Path to a pretrained PHOCNet binary protofile.')
    qbs_parser.add_argument('--train_xml_file', action='store', type=str, required=True,
                            help='The READ-style XML file of words used in training')
    qbs_parser.add_argument('--test_xml_file', action='store', type=str, required=True,
                            help='The READ-style XML file of words to be used for testing')
    qbs_parser.add_argument('--annotation_delimiter', '-ad', action='store', type=str,
                            help='If the annotation in the XML is separated by special delimiters, it can be specified here.')
    qbs_parser.add_argument('--doc_img_dir', action='store', type=str, required=True,
                            help='The location of the document images.')  
    qbs_parser.add_argument('--use_lower_case_only', '-ulco', action='store_true', default=False,
                            help='Flag indicating to convert all annotations from the XML to lower case before proceeding')
    qbs_parser.add_argument('--gpu_id', '-gpu', action='store', type=int, 
                            help='The ID of the GPU to use. If not specified, prediction is run in CPU mode.')
    qbs_parser.set_defaults(func=eval.eval_qbs)
    
    # --- qbe
    qbe_parser.add_argument('--phocnet_bin_path', action='store', type=str, required=True,
                            help='Path to a pretrained PHOCNet binary protofile.')
    qbe_parser.add_argument('--word_xml_file', action='store', type=str, required=True,
                            help='The READ-style XML file of words for which to run the QbE evaluation')
    qbe_parser.add_argument('--doc_img_dir', action='store', type=str, required=True,
                            help='The location of the document images.')  
    qbe_parser.add_argument('--gpu_id', '-gpu', action='store', type=int, 
                            help='The ID of the GPU to use. If not specified, prediction is run in CPU mode.')
    qbe_parser.add_argument('--metric', action='store', type=str, default='braycurtis',
                            help='The metric to be used when comparing the PHOCNet output. Possible: all scipy metrics. Default: braycurtis')
    qbe_parser.add_argument('--debug_mode', '-dm', action='store_true', default=False,
                            help='Flag indicating to run the PHOCNet training in debug mode.')
    qbe_parser.add_argument('--deploy_proto_path', '-dpp', action='store', type=str, default='/tmp/deploy_phocnet.prototxt',
                            help='Location where to save the deploy file. Default: /tmp/deploy_phocnet.prototxt')
    qbe_parser.add_argument('--phoc_size', '-ps', action='store', type=int, default=540,
                            help='The size of the PHOC to be used. Default: 540')
    qbe_parser.set_defaults(func=eval.eval_qbe)
    
    # --- sliding-window
    sliding_window_parser.add_argument('--phocnet_bin_path', action='store', type=str, required=True,
                                       help='Path to a pretrained PHOCNet binary protofile.')
    sliding_window_parser.add_argument('--word_xml_file', action='store', type=str, required=True,
                                       help='The READ-style XML file of words for which to predict the PHOCs')
    sliding_window_parser.add_argument('--doc_img_dir', action='store', type=str, required=True,
                                       help='The location of the document images.')  
    sliding_window_parser.add_argument('--gpu_id', '-gpu', action='store', type=int, 
                                       help='The ID of the GPU to use. If not specified, prediction is run in CPU mode.')
    sliding_window_parser.add_argument('--frame_width', '-fw', action='store', type=int, required=True,
                                       help='The window width of the sliding window')
    sliding_window_parser.add_argument('--step_size', '-ss', action='store', type=int, required=True,
                                       help='The step size for the sliding window')
    sliding_window_parser.add_argument('--no_padding', action='store_true', default=False,
                                       help='Flag indicating to switch of the padding of half the frame width.')
    sliding_window_parser.add_argument('--debug_mode', '-dm', action='store_true', default=False,
                                       help='Flag indicating to run the PHOCNet training in debug mode.')
    sliding_window_parser.add_argument('--phoc_size', '-ps', action='store', type=int, default=540,
                                       help='The size of the PHOC to be used. Default: 540')
    sliding_window_parser.add_argument('--output_dir', '-od', action='store', type=str, default='.',
                                       help='Location where to save the estimated PHOCs. Default: working directory')
    sliding_window_parser.add_argument('--deploy_proto_path', '-dpp', action='store', type=str, default='/tmp/deploy_phocnet.prototxt',
                                       help='Location where to save the deploy file. Default: /tmp/deploy_phocnet.prototxt')
    sliding_window_parser.set_defaults(func=eval.predict_and_save_sliding_window)    
           
    # run everything   
    subfunc = parser.parse_args()
    args = vars(subfunc).copy()
    del args['func']
    subfunc.func(**args)